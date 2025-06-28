# ruff: noqa: S311
from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from affctrllib import PTP, AffComm, AffPosCtrl, AffStateThread, Logger, Timer
from pyplotutil.datautil import Data
from scipy import interpolate

from affetto_nn_ctrl._typing import NoDefault, no_default
from affetto_nn_ctrl.event_logging import event_logger
from affetto_nn_ctrl.random_utility import get_rng

if sys.version_info < (3, 11):
    import tomli as tomllib  # type: ignore[reportMissingImports]
else:
    import tomllib

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from numpy.random import Generator

    from affetto_nn_ctrl import CONTROLLER_T, RefFuncType


MIN_UPDATE_Q_DELTA = 1e-4
WAIST_JOINT_INDEX = 0
WAIST_JOINT_LIMIT = (40.0, 60.0)


def create_controller(
    config: str,
    sfreq: float | None,
    cfreq: float | None,
    waiting_time: float = 0.0,
) -> CONTROLLER_T:
    event_logger().info("Loaded config: %s", config)
    event_logger().debug("sensor frequency: %s, control frequency: %s", sfreq, cfreq)

    comm = AffComm(config_path=config)
    comm.create_command_socket()
    state = AffStateThread(config=config, freq=sfreq, logging=False, output=None, butterworth=True)
    ctrl = AffPosCtrl(config_path=config, freq=cfreq)
    state.prepare()
    state.start()
    event_logger().debug("Controller created.")

    if waiting_time > 0:
        event_logger().info("Waiting until robot gets stationary for %s s...", waiting_time)
        time.sleep(waiting_time)

    return comm, ctrl, state


def create_default_logger(dof: int) -> Logger:
    logger = Logger()
    logger.set_labels(
        "t",
        # raw data
        [f"rq{i}" for i in range(dof)],
        [f"rdq{i}" for i in range(dof)],
        [f"rpa{i}" for i in range(dof)],
        [f"rpb{i}" for i in range(dof)],
        # estimated states
        [f"q{i}" for i in range(dof)],
        [f"dq{i}" for i in range(dof)],
        [f"pa{i}" for i in range(dof)],
        [f"pb{i}" for i in range(dof)],
        # command data
        [f"ca{i}" for i in range(dof)],
        [f"cb{i}" for i in range(dof)],
        [f"qdes{i}" for i in range(dof)],
        [f"dqdes{i}" for i in range(dof)],
    )
    event_logger().debug("Default logger created.")

    return logger


def create_const_trajectory(
    qdes: float | list[float] | np.ndarray,
    joint: int | list[int],
    q0: np.ndarray,
) -> tuple[RefFuncType, RefFuncType]:
    def qdes_func(_: float) -> np.ndarray:
        q = np.copy(q0)
        q[0] = 50  # make waist joint keep at middle.
        q[joint] = qdes
        return q

    def dqdes_func(_: float) -> np.ndarray:
        return np.zeros(len(q0), dtype=float)

    return qdes_func, dqdes_func


def reset_logger(logger: Logger | None, log_filename: str | Path | None) -> Logger | None:
    if logger is not None:
        logger.erase_data()
        event_logger().debug("Logger data has been erased.")
        if log_filename is not None:
            logger.set_filename(log_filename)
            event_logger().debug("Logger filename is updated: %s", log_filename)
    return logger


def select_time_updater(timer: Timer, time_updater: str) -> Callable[[], float]:
    current_time_func: Callable[[], float]
    if time_updater == "elapsed":
        current_time_func = timer.elapsed_time
    elif time_updater == "accumulated":
        current_time_func = timer.accumulated_time
    else:
        msg = f"unrecognized time updater: {time_updater}"
        raise ValueError(msg)
    return current_time_func


def control_position(
    controller: CONTROLLER_T,
    qdes_func: Callable[[float], np.ndarray | float],
    dqdes_func: Callable[[float], np.ndarray | float],
    duration: float,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
    time_updater: str = "elapsed",
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    reset_logger(logger, log_filename)
    comm, ctrl, state = controller
    ca, cb = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    timer = Timer(rate=ctrl.freq)
    current_time = select_time_updater(timer, time_updater)

    timer.start()
    t = 0.0
    while t < duration:
        sys.stdout.write(f"\r{header_text} [{t:6.2f}/{duration:.2f}]")
        t = current_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        qdes = qdes_func(t)
        dqdes = dqdes_func(t)
        ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes)
        timer.block()
    sys.stdout.write("\n")
    # Return the last commands that have been sent to the valve.
    return ca, cb


def record_motion(
    active_joints: list[int],
    q0: np.ndarray,
    controller: CONTROLLER_T,
    duration: float,
    q_limit: tuple[float, float] | list[tuple[float, float]] | None,
    logger: Logger,
    data_file_path: Path,
    time_updater: str = "elapsed",
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    reset_logger(logger, data_file_path)
    comm, ctrl, state = controller
    ca, cb = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    timer = Timer(rate=ctrl.freq)
    current_time = select_time_updater(timer, time_updater)

    # Make active joints inactive in controller.
    ctrl.set_inactive_joints(active_joints, pressure=0.0)
    # Let the controller keep the initial position.
    qdes = q0.copy()
    dqdes = np.zeros(ctrl.dof, dtype=float)

    # Make limit of joint positions when recording.
    if isinstance(q_limit, tuple):
        x = (min(q_limit), max(q_limit))
        q_limit = [x for _ in active_joints]

    timer.start()
    t = 0.0
    while t < duration:
        sys.stdout.write(f"\r{header_text} [{t:6.2f}/{duration:.2f}]")
        t = current_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        # Clip measured position between specified limits.
        if q_limit:
            q[active_joints] = [min(q_limit[i][1], max(q_limit[i][0], x)) for i, x in enumerate(q[active_joints])]
        ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes)
        timer.block()
    sys.stdout.write("\n")
    # Return the last commands that have been sent to the valve.
    return ca, cb


def control_pressure(
    controller: CONTROLLER_T,
    ca_func: Callable[[float], np.ndarray | float],
    cb_func: Callable[[float], np.ndarray | float],
    duration: float,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
    time_updater: str = "elapsed",
    header_text: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    reset_logger(logger, log_filename)
    comm, ctrl, state = controller
    ca, cb = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    timer = Timer(rate=ctrl.freq)
    dummy = np.asarray([-1.0 for _ in range(ctrl.dof)])
    current_time = select_time_updater(timer, time_updater)

    timer.start()
    t = 0.0
    while t < duration:
        sys.stdout.write(f"\r{header_text} [{t:6.2f}/{duration:.2f}]")
        t = current_time()
        rq, rdq, rpa, rpb = state.get_raw_states()
        q, dq, pa, pb = state.get_states()
        ca = np.asarray(ca_func(t))
        cb = np.asarray(cb_func(t))
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, dummy, dummy)
        timer.block()
    sys.stdout.write("\n")
    # Return the last commands that have been sent to the valve.
    return ca, cb


def get_back_home_position(
    controller: CONTROLLER_T,
    q_home: np.ndarray,
    duration: float,
    duration_keep_steady: float = 0.0,
    header_text: str = "Getting back to home position...",
) -> tuple[np.ndarray, np.ndarray]:
    comm, ctrl, state = controller
    q0 = state.q
    ptp = PTP(q0, q_home, duration)
    total_duration = duration + duration_keep_steady
    qdes_func, dqdes_func = ptp.q, ptp.dq
    event_logger().debug(header_text)
    event_logger().debug("  duration: %s, total: %s", duration, total_duration)
    event_logger().debug("  q_home  : %s", q_home)
    ca, cb = control_position(controller, qdes_func, dqdes_func, total_duration, header_text=header_text)
    event_logger().debug("Done")
    return ca, cb


def get_back_home_pressure(
    controller: CONTROLLER_T,
    ca_home: np.ndarray,
    cb_home: np.ndarray,
    duration: float,
    duration_keep_steady: float = 0.0,
    header_text: str = "Getting back to home position (by valve)...",
) -> tuple[np.ndarray, np.ndarray]:
    comm, ctrl, state = controller
    ca0, cb0 = np.zeros(ctrl.dof, dtype=float), np.zeros(ctrl.dof, dtype=float)
    ptp_ca = PTP(ca0, ca_home, duration, profile_name="const")
    ptp_cb = PTP(cb0, cb_home, duration, profile_name="const")
    total_duration = duration + duration_keep_steady
    ca_func, cb_func = ptp_ca.q, ptp_cb.q
    event_logger().debug(header_text)
    event_logger().debug("  duration: %s, total: %s", duration, total_duration)
    event_logger().debug("  ca_home : %s", ca_home)
    event_logger().debug("  cb_home : %s", cb_home)
    ca, cb = control_pressure(controller, ca_func, cb_func, total_duration, header_text=header_text)
    event_logger().debug("Done")
    return ca, cb


ERR_MSG_RESOLVE_JOINTS_STR = "Unable to resolve given joints string"


def _resolve_single_digit(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        msg = f"{ERR_MSG_RESOLVE_JOINTS_STR}: {s}"
        raise ValueError(msg) from None


def _resolve_consecutive_digits(s: str) -> list[int]:
    digits = s.split("-")
    if len(digits) > 1:
        try:
            beg = int(digits[0])
            end = int(digits[-1])
            return list(range(beg, end + 1))
        except ValueError:
            pass
    msg = f"{ERR_MSG_RESOLVE_JOINTS_STR}: {s}"
    raise ValueError(msg)


def _split_joints_str(s: str, delim: str) -> Iterable[str]:
    return re.split(delim, s)


def _resolve_joints_str(joints_str: str, dof: int | None) -> list[int]:
    resolved_joints: list[int] = []
    for s in _split_joints_str(joints_str, r"[\s,]+"):
        if len(s) == 0:
            # Skip for zero-length string
            continue

        if s.isdigit():
            # Resolve single digit string
            resolved_joints.append(_resolve_single_digit(s))

        elif s == "all":
            # Resolve all joints
            if dof is not None:
                resolved_joints.extend(range(dof))
            else:
                msg = f"{ERR_MSG_RESOLVE_JOINTS_STR}: {s}"
                msg += ", Hint: provide an optional DOF argument"
                raise ValueError(msg)

        elif "-" in s:
            # Resolve consecutive digits string
            resolved_joints.extend(_resolve_consecutive_digits(s))

        else:
            msg = f"{ERR_MSG_RESOLVE_JOINTS_STR}: {s}"
            raise ValueError(msg)

    return resolved_joints


def _sort_and_remove_duplicates(resolved_joints: list[int]) -> list[int]:
    return sorted(set(resolved_joints))


def resolve_joints_str(joints_str: str | Iterable[str] | None, dof: int | None = None) -> list[int]:
    if joints_str is None:
        joints_str = ["all"]
    elif isinstance(joints_str, str):
        joints_str = [joints_str]

    resolved_joints: list[int] = []
    for s in joints_str:
        resolved_joints.extend(_resolve_joints_str(s, dof))

    return _sort_and_remove_duplicates(resolved_joints)


class RobotInitializer:
    _dof: int
    _duration: float
    _duration_keep_steady: float
    _manner: str
    _q_init: np.ndarray
    _ca_init: np.ndarray
    _cb_init: np.ndarray

    DEFAULT_DURATION = 5.0
    DEFAULT_DURATION_KEEP_STEADY = 5.0
    DEFAULT_MANNER = "position"
    DEFAULT_Q_INIT = 50.0
    DEFAULT_CA_INIT = 0.0
    DEFAULT_CB_INIT = 120.0

    def __init__(
        self,
        dof: int,
        *,
        config: str | Path | None = None,
        duration: float | None = None,
        duration_keep_steady: float | None = None,
        manner: str | None = None,
        q_init: Sequence[float] | np.ndarray | float | None = None,
        ca_init: Sequence[float] | np.ndarray | float | None = None,
        cb_init: Sequence[float] | np.ndarray | float | None = None,
    ) -> None:
        self._dof = dof
        # Set default values
        self.duration = self.DEFAULT_DURATION
        self.duration_keep_steady = self.DEFAULT_DURATION_KEEP_STEADY
        self.set_manner(self.DEFAULT_MANNER)
        self.set_q_init(self.DEFAULT_Q_INIT)
        self.set_ca_init(self.DEFAULT_CA_INIT)
        self.set_cb_init(self.DEFAULT_CB_INIT)
        # Update values based on config file
        if config is not None:
            self.load_config(config)
        # Update values based on arguments
        self._update_values(duration, duration_keep_steady, manner, q_init, ca_init, cb_init)

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        self._duration = duration

    @property
    def duration_keep_steady(self) -> float:
        return self._duration_keep_steady

    @duration_keep_steady.setter
    def duration_keep_steady(self, duration_keep_steady: float) -> None:
        self._duration_keep_steady = duration_keep_steady

    @property
    def total_duration(self) -> float:
        return self.duration + self.duration_keep_steady

    def get_manner(self) -> str:
        return self._manner

    def set_manner(self, manner: str) -> str:
        match manner:
            case "position" | "pos" | "p" | "q":
                self._manner = "position"
            case "pressure" | "pres" | "pre" | "valve" | "v":
                self._manner = "pressure"
            case _:
                msg = f"Unrecognized manner for RobotInitializer: {manner}"
                raise ValueError(msg)
        return self._manner

    @staticmethod
    def normalize_array(dof: int, given_value: Sequence[float] | np.ndarray | float) -> np.ndarray:
        if isinstance(given_value, float | int):
            array = np.full((dof,), given_value, dtype=float)
        elif len(given_value) == dof:
            array = np.array(given_value, dtype=float)
        elif len(given_value) == 1:
            array = np.full((dof,), given_value[0], dtype=float)
        else:
            msg = f"Unable to set values due to size mismatch: dof={dof}, given_value={given_value}"
            raise ValueError(msg)
        return array

    def get_q_init(self) -> np.ndarray:
        return self._q_init

    def set_q_init(self, q_init: Sequence[float] | np.ndarray | float) -> np.ndarray:
        self._q_init = self.normalize_array(self.dof, q_init)
        return self._q_init

    def get_ca_init(self) -> np.ndarray:
        return self._ca_init

    def set_ca_init(self, ca_init: Sequence[float] | np.ndarray | float) -> np.ndarray:
        self._ca_init = self.normalize_array(self.dof, ca_init)
        return self._ca_init

    def get_cb_init(self) -> np.ndarray:
        return self._cb_init

    def set_cb_init(self, cb_init: Sequence[float] | np.ndarray | float) -> np.ndarray:
        self._cb_init = self.normalize_array(self.dof, cb_init)
        return self._cb_init

    def _update_values(
        self,
        duration: float | None,
        duration_keep_steady: float | None,
        manner: str | None,
        q_init: Sequence[float] | np.ndarray | float | None,
        ca_init: Sequence[float] | np.ndarray | float | None,
        cb_init: Sequence[float] | np.ndarray | float | None,
    ) -> None:
        if duration is not None:
            self.duration = duration
        if duration_keep_steady is not None:
            self.duration_keep_steady = duration_keep_steady
        if manner is not None:
            self.set_manner(manner)
        if q_init is not None:
            self.set_q_init(q_init)
        if ca_init is not None:
            self.set_ca_init(ca_init)
        if cb_init is not None:
            self.set_cb_init(cb_init)

    def load_config(self, config: str | Path) -> None:
        with Path(config).open("rb") as f:
            c = tomllib.load(f)
        affetto_config = c["affetto"]
        init_config = affetto_config.get("init", None)
        if init_config is None:
            return
        duration = init_config.get("duration", None)
        duration_keep_steady = init_config.get("duration_keep_steady", None)
        manner = init_config.get("manner", None)
        q_init = init_config.get("q", None)
        ca_init = init_config.get("ca", None)
        cb_init = init_config.get("cb", None)
        self._update_values(duration, duration_keep_steady, manner, q_init, ca_init, cb_init)

    def get_back_home(
        self,
        controller: CONTROLLER_T,
        duration: float | None = None,
        duration_keep_steady: float | None = None,
        header_text: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if duration is None:
            duration = self.duration
        if duration_keep_steady is None:
            duration_keep_steady = self.duration_keep_steady
        if header_text is None:
            header_text = f"Getting back to home position (by {self.get_manner()})..."
        if self.get_manner() == "pressure":
            ca, cb = get_back_home_pressure(
                controller,
                self.get_ca_init(),
                self.get_cb_init(),
                duration,
                duration_keep_steady,
                header_text=header_text,
            )
        else:
            ca, cb = get_back_home_position(
                controller,
                self.get_q_init(),
                duration,
                duration_keep_steady,
                header_text=header_text,
            )
        return ca, cb


def release_pressure(
    controller: CONTROLLER_T,
    duration: float = 1.0,
    duration_keep_steady: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    initializer = RobotInitializer(
        controller[1].dof,
        duration=duration,
        duration_keep_steady=duration_keep_steady,
        manner="pressure",
        ca_init=0.0,
        cb_init=0.0,
    )
    return initializer.get_back_home(controller, header_text="Releasing pressure...")


class RandomTrajectory:
    active_joints: list[int]
    t0: float
    q0: np.ndarray
    update_t_range_list: list[tuple[float, float]]
    update_q_range_list: list[tuple[float, float]]
    update_q_limit_list: list[tuple[float, float]]
    update_profile: str
    async_update: bool
    rng: Generator
    sync_updater: PTP
    async_updater: list[PTP]

    def __init__(
        self,
        active_joints: list[int],
        t0: float,
        q0: np.ndarray,
        update_t_range: tuple[float, float] | list[tuple[float, float]],
        update_q_range: tuple[float, float] | list[tuple[float, float]],
        update_q_limit: tuple[float, float] | list[tuple[float, float]],
        update_profile: str = "trapezoidal",
        seed: int | NoDefault | None = no_default,
        *,
        async_update: bool = False,
    ) -> None:
        self.active_joints = active_joints
        self.t0 = t0
        self.q0 = q0.copy()
        self.set_update_t_range(active_joints, update_t_range)
        self.set_update_q_range(active_joints, update_q_range)
        self.set_update_q_limit(active_joints, update_q_limit)
        self.update_profile = update_profile
        self.async_update = async_update
        self.rng = get_rng(seed)
        self.reset_updater()

    @staticmethod
    def get_list_of_range(
        active_joints: list[int],
        given_range: tuple[float, float] | list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if isinstance(given_range, tuple):
            x = (min(given_range), max(given_range))
            range_list = [x for _ in active_joints]
        elif len(active_joints) == len(given_range):
            range_list = [(min(x), max(x)) for x in given_range]
        else:
            msg = (
                "Lengths of lists (active joints / range list) are mismatch: "
                f"{len(active_joints)} vs {len(given_range)}"
            )
            raise ValueError(msg)
        return range_list

    def set_update_t_range(
        self,
        active_joints: list[int],
        update_t_range: tuple[float, float] | list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        self.update_t_range_list = self.get_list_of_range(active_joints, update_t_range)
        return self.update_t_range_list

    def set_update_q_range(
        self,
        active_joints: list[int],
        update_q_range: tuple[float, float] | list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        self.update_q_range_list = self.get_list_of_range(active_joints, update_q_range)
        return self.update_q_range_list

    def set_update_q_limit(
        self,
        active_joints: list[int],
        update_q_limit: tuple[float, float] | list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        self.update_q_limit_list = self.get_list_of_range(active_joints, update_q_limit)
        if isinstance(update_q_limit, tuple) and WAIST_JOINT_INDEX in active_joints:
            # When update_q_limit is given as tuple and waist joint is included in active
            # joints list, reduce limits of the waist joint.
            waist_index = active_joints.index(WAIST_JOINT_INDEX)
            self.update_q_limit_list[waist_index] = WAIST_JOINT_LIMIT
        return self.update_q_limit_list

    def generate_new_duration(self, t_range: tuple[float, float]) -> float:
        return self.rng.uniform(min(t_range), max(t_range))

    def generate_new_position(
        self,
        q0: float,
        q_range: tuple[float, float],
        q_limit: tuple[float, float],
        min_update_q_delta: float = MIN_UPDATE_Q_DELTA,
    ) -> float:
        dmin, dmax = min(q_range), max(q_range)
        qmin, qmax = min(q_limit), max(q_limit)
        qdes = q0
        ok = False
        while not ok:
            delta = self.rng.uniform(dmin, dmax)
            qdes = q0 + self.rng.choice([-1, 1]) * delta
            if qdes < qmin:
                qdes = qmin + (qmin - qdes)
            elif qdes > qmax:
                qdes = qmax - (qdes - qmax)
            qdes = max(min(qmax, qdes), qmin)
            qdiff = abs(qdes - q0)
            if qdiff > min_update_q_delta:
                ok = True
        return qdes

    def initialize_sync_updater(self, t0: float, q0: np.ndarray) -> PTP:
        if not all(x == self.update_t_range_list[0] for x in self.update_t_range_list):
            msg = "Enabled sync update but various update t range is given."
            warnings.warn(msg, stacklevel=2)

        active_q0 = q0[self.active_joints]
        duration = self.generate_new_duration(self.update_t_range_list[0])
        qdes = np.array(
            [
                self.generate_new_position(active_q0[i], self.update_q_range_list[i], self.update_q_limit_list[i])
                for i in range(len(self.active_joints))
            ],
        )
        return PTP(active_q0, qdes, duration, t0, profile_name=self.update_profile)

    def initialize_async_updater(self, t0: float, q0: np.ndarray) -> list[PTP]:
        ptp_list: list[PTP] = []
        for i, j in enumerate(self.active_joints):
            active_q0 = q0[j]
            duration = self.generate_new_duration(self.update_t_range_list[i])
            qdes = self.generate_new_position(active_q0, self.update_q_range_list[i], self.update_q_limit_list[i])
            ptp_list.append(PTP(active_q0, qdes, duration, t0, profile_name=self.update_profile))
        return ptp_list

    def initialize_updater(self, t0: float, q0: np.ndarray) -> None:
        if self.async_update:
            self.async_updater = self.initialize_async_updater(t0, q0)
        else:
            self.sync_updater = self.initialize_sync_updater(t0, q0)

    def update_sync_updater(self, t: float) -> None:
        ptp = self.sync_updater
        if ptp.t0 + ptp.T < t:
            new_t0 = ptp.t0 + ptp.T
            new_q0 = ptp.qF
            new_duration = self.generate_new_duration(self.update_t_range_list[0])
            new_qdes = np.array(
                [
                    self.generate_new_position(new_q0[i], self.update_q_range_list[i], self.update_q_limit_list[i])
                    for i in range(len(self.active_joints))
                ],
            )
            new_ptp = PTP(new_q0, new_qdes, new_duration, new_t0, profile_name=self.update_profile)
            self.sync_updater = new_ptp

    def update_async_updater(self, t: float) -> None:
        for i, ptp in enumerate(self.async_updater):
            if ptp.t0 + ptp.T < t:
                new_t0 = ptp.t0 + ptp.T
                new_q0 = ptp.qF
                new_duration = self.generate_new_duration(self.update_t_range_list[i])
                new_qdes = self.generate_new_position(new_q0, self.update_q_range_list[i], self.update_q_limit_list[i])
                new_ptp = PTP(new_q0, new_qdes, new_duration, new_t0, profile_name=self.update_profile)
                self.async_updater[i] = new_ptp

    def reset_updater(self, t0: float | None = None, q0: np.ndarray | None = None) -> None:
        if t0 is not None:
            self.t0 = t0
        else:
            t0 = self.t0

        if q0 is not None:
            self.q0 = q0
        else:
            q0 = self.q0

        self.initialize_updater(t0, q0)

    def get_qdes_func(self) -> RefFuncType:
        def qdes_async(t: float) -> np.ndarray:
            self.update_async_updater(t)
            qdes = self.q0.copy()
            qdes[self.active_joints] = [ptp.q(t) for ptp in self.async_updater]
            return qdes

        def qdes_sync(t: float) -> np.ndarray:
            self.update_sync_updater(t)
            qdes = self.q0.copy()
            qdes[self.active_joints] = self.sync_updater.q(t)
            return qdes

        if self.async_update:
            return qdes_async
        return qdes_sync

    def get_dqdes_func(self) -> RefFuncType:
        def dqdes_async(t: float) -> np.ndarray:
            self.update_async_updater(t)
            dqdes = np.zeros(self.q0.shape, dtype=float)
            dqdes[self.active_joints] = [ptp.dq(t) for ptp in self.async_updater]
            return dqdes

        def dqdes_sync(t: float) -> np.ndarray:
            self.update_sync_updater(t)
            dqdes = np.zeros(self.q0.shape, dtype=float)
            dqdes[self.active_joints] = self.sync_updater.dq(t)
            return dqdes

        if self.async_update:
            return dqdes_async
        return dqdes_sync


class Spline:
    active_joints: list[int]
    _data: Data
    _dof: int
    _q0: np.ndarray
    _duration: float
    _use_filter_value: bool

    def __init__(
        self,
        data: str | Path | Data,
        active_joints: list[int],
        s: float | None = None,
        *,
        use_filter_value: bool = False,
    ) -> None:
        self.active_joints = active_joints
        self._use_filter_value = use_filter_value
        if isinstance(data, str | Path):
            self.load_data(data)
        else:
            self.set_data(data)
        self.interpolate(s)

    @property
    def data(self) -> Data:
        return self._data

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def q0(self) -> np.ndarray:
        return self._q0

    @property
    def duration(self) -> float:
        return self._duration

    @staticmethod
    def count_dof(data: Data) -> int:
        pattern = re.compile(r"rq\d+")
        array = data.dataframe.columns.array
        return len(list(filter(pattern.match, array)))

    def _set_data(self) -> None:
        self._dof = self.count_dof(self._data)
        if self._use_filter_value:
            self._q0 = np.array([getattr(self._data, f"q{i}")[0] for i in range(self._dof)])
        else:
            self._q0 = np.array([getattr(self._data, f"rq{i}")[0] for i in range(self._dof)])
        self._duration = self._data.t.to_numpy()[-1]

    def load_data(self, datapath: str | Path) -> Data:
        self._data = Data(datapath)
        self._set_data()
        return self.data

    def set_data(self, data: Data) -> Data:
        self._data = data
        self._set_data()
        return self.data

    def interpolate(self, s: float | None) -> None:
        if s is None:
            m = len(self._data)
            s = m - np.sqrt(2.0 * m)
        x = self._data.t
        self._tck = []
        self._der = []
        for i in self.active_joints:
            if self._use_filter_value:
                y = getattr(self._data, f"q{i}")
            else:
                y = getattr(self._data, f"rq{i}")
            tck = interpolate.splrep(x, y, s=s)
            self._tck.append(tck)
            self._der.append(interpolate.splder(tck))

    def get_qdes_func(
        self,
        q0: np.ndarray | None = None,
        duration: float | None = None,
    ) -> RefFuncType:
        if q0 is None:
            q0 = self.q0
        if duration is None:
            duration = self.duration

        def qdes_func(t: float) -> np.ndarray:
            q = q0.copy()
            q[self.active_joints] = [interpolate.splev(t, tck) for tck in self._tck]
            return q

        return qdes_func

    def get_dqdes_func(
        self,
        q0: np.ndarray | None = None,
        duration: float | None = None,
    ) -> RefFuncType:
        if q0 is None:
            q0 = self.q0
        if duration is None:
            duration = self.duration

        def dqdes_func(t: float) -> np.ndarray:
            dq = np.zeros(q0.shape, dtype=float)
            dq[self.active_joints] = [interpolate.splev(t, der) for der in self._der]
            return dq

        return dqdes_func


# Local Variables:
# jinx-local-words: "cb const dT dof dq dqdes init pb pos qdes rb rdq rpa rpb rq"
# End:
