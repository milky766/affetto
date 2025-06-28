from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, TypedDict, TypeVar, cast

import joblib
import numpy as np
import pandas as pd
from affctrllib import Logger, Timer
from pyplotutil.datautil import Data
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from affetto_nn_ctrl import RefFuncType
from affetto_nn_ctrl.control_utility import reset_logger, select_time_updater
from affetto_nn_ctrl.event_logging import event_logger

if TYPE_CHECKING:
    from affctrllib import AffPosCtrl

    from affetto_nn_ctrl import CONTROLLER_T
    from affetto_nn_ctrl._typing import T, Unknown


from sklearn.metrics import r2_score

if sys.version_info >= (3, 11):
    from typing import NotRequired
    

    import tomllib
else:
    import tomli as tomllib  # type: ignore[reportMissingImports]
    from typing_extensions import NotRequired


@dataclass
class DataAdapterParamsBase:
    pass


class StatesBase(TypedDict):
    pass


class RefsBase(TypedDict):
    pass


class InputsBase(TypedDict):
    pass


DataAdapterParamsType = TypeVar("DataAdapterParamsType", bound=DataAdapterParamsBase)
StatesType = TypeVar("StatesType", bound=StatesBase)
RefsType = TypeVar("RefsType", bound=RefsBase)
InputsType = TypeVar("InputsType", bound=InputsBase)


class DefaultStates(StatesBase):
    q: np.ndarray
    dq: np.ndarray
    ddq: NotRequired[np.ndarray]
    pa: np.ndarray
    pb: np.ndarray


class DefaultRefs(RefsBase):
    qdes: RefFuncType
    dqdes: RefFuncType


class DefaultInputs(InputsBase):
    ca: np.ndarray
    cb: np.ndarray


class DataAdapterBase(ABC, Generic[DataAdapterParamsType, StatesType, RefsType, InputsType]):
    _params: DataAdapterParamsType

    def __init__(self, params: DataAdapterParamsType) -> None:
        self._params = params

    @property
    def params(self) -> DataAdapterParamsType:
        return self._params

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"

    @abstractmethod
    def make_feature(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def make_target(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def make_model_input(self, t: float, states: StatesType, refs: RefsType) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def make_ctrl_input(self, y: np.ndarray, base_inputs: InputsType) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class DummyDataAdapter(DataAdapterBase[DataAdapterParamsBase, StatesBase, RefsBase, InputsBase]):
    def make_feature(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_target(self, dataset: Data) -> np.ndarray:
        raise NotImplementedError

    def make_model_input(self, t: float, states: StatesBase, refs: RefsBase) -> np.ndarray:
        raise NotImplementedError

    def make_ctrl_input(self, y: np.ndarray, base_inputs: InputsBase) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


dummy_data_adapter = DummyDataAdapter(DataAdapterParamsBase())


def _get_keys(symbols: Iterable[str], active_joints: list[int], *, add_t: bool = False) -> list[str]:
    keys: list[str] = []
    if add_t:
        keys.append("t")
    for s in symbols:
        keys.extend([f"{s}{i}" for i in active_joints])
    return keys

@dataclass
class SingleShotMultiHorizonParams(DataAdapterParamsBase):
    """Parameters for the single-shot multi-horizon handler."""
    active_joints: list[int]
    dt: float
    ctrl_step: int
    min_preview_step: int
    max_preview_step: int

class SingleShotMultiHorizonHandler(DataAdapterBase[SingleShotMultiHorizonParams, DefaultStates, DefaultRefs, DefaultInputs]):
    """
    Generates a dataset for single-shot multi-horizon learning.
    Input X: [current_state, future_reference_trajectory]
    Output Y: [future_control_commands]
    """
    def make_feature(self, dataset: Data) -> np.ndarray:
        # 入力Xを生成
        joints = self.params.active_joints
        N_min, N_max = self.params.min_preview_step, self.params.max_preview_step

        state_keys = _get_keys(["q", "dq", "pa", "pb"], joints)
        ref_keys = _get_keys(["q"], joints)

        # 有効なデータ範囲を計算
        valid_len = len(dataset.df) - N_max

        # 現在の状態 s_t
        s_t = dataset.df.loc[:valid_len-1, state_keys].values

        # 未来の参照軌道 r_{t+N_min} ... r_{t+N_max}
        future_refs_list = []
        for k in range(N_min, N_max + 1):
            ref_at_k = dataset.df.loc[k:valid_len+k-1, ref_keys].values
            future_refs_list.append(ref_at_k)

        future_refs_concat = np.concatenate(future_refs_list, axis=1)

        # s_t と未来の参照軌道を連結してXを作成
        X = np.concatenate([s_t, future_refs_concat], axis=1)
        return X

    def make_target(self, dataset: Data) -> np.ndarray:
        # 正解ラベルYを生成
        joints = self.params.active_joints
        N_min, N_max = self.params.min_preview_step, self.params.max_preview_step

        ctrl_keys = _get_keys(["ca", "cb"], joints)

        # 有効なデータ範囲を計算
        valid_len = len(dataset.df) - N_max

        # 未来の制御指令 u_{t+N_min} ... u_{t+N_max}
        future_ctrls_list = []
        for k in range(N_min, N_max + 1):
            ctrl_at_k = dataset.df.loc[k:valid_len+k-1, ctrl_keys].values
            future_ctrls_list.append(ctrl_at_k)

        # 全ての未来の制御指令を連結してYを作成
        Y = np.concatenate(future_ctrls_list, axis=1)
        return Y

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        raise NotImplementedError
    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        raise NotImplementedError
    def reset(self) -> None:
        pass

@dataclass
class PreviewRefParams(DataAdapterParamsBase):
    active_joints: list[int]
    dt: float
    ctrl_step: int
    preview_step: int
    include_dqdes: bool = False


class PreviewRef(DataAdapterBase[PreviewRefParams, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(self, params: PreviewRefParams) -> None:
        super().__init__(params)
        if params.ctrl_step < 1:
            msg = f"DelayStatesParams.ctrl_step must be larger than or equal to 1: {params.ctrl_step}"
            raise ValueError(msg)
        if params.preview_step < 0:
            msg = f"DelayStatesParams.delay_step must be larger than or equal to 0: {params.preview_step}"
            raise ValueError(msg)
        self.reset()

    def make_feature(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        preview_step = self.params.preview_step
        shift = ctrl_step + preview_step
        states = extract_data(dataset, _get_keys(["q", "dq", "pa", "pb"], joints), end=-shift)
        ref_keys: list[str] = ["q"]
        if self.params.include_dqdes:
            ref_keys.append("dq")
        reference = extract_data(
            dataset,
            _get_keys(ref_keys, joints),
            start=shift,
            keys_replace=_get_keys([f"{x}des" for x in ref_keys], joints),
        )
        feature_data = pd.concat((states, reference), axis=1)
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        preview_step = self.params.preview_step
        shift = ctrl_step + preview_step
        ctrl_input = extract_data(dataset, _get_keys(["ca", "cb"], joints), end=-shift)
        return ctrl_input.to_numpy()

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        joints = self.params.active_joints
        preview_time = self.params.preview_step * self.params.dt
        q = states["q"][joints]
        dq = states["dq"][joints]
        pa = states["pa"][joints]
        pb = states["pb"][joints]
        qdes = refs["qdes"](t + preview_time)[joints]
        if self.params.include_dqdes:
            dqdes = refs["dqdes"](t + preview_time)[joints]
            model_input = np.concatenate((q, dq, pa, pb, qdes, dqdes))
        else:
            model_input = np.concatenate((q, dq, pa, pb, qdes))  # concatenate feature vectors horizontally
        # Make matrix that has 1 row and n_features columns.
        return np.atleast_2d(model_input)

    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        # `y` has has 1 row and n_targets columns.
        joints = self.params.active_joints
        y = np.ravel(y)
        ca, cb = base_inputs["ca"], base_inputs["cb"]
        n = len(joints)
        ca[joints] = y[:n]
        cb[joints] = y[n:]
        return ca, cb

    def reset(self) -> None:
        pass


@dataclass
class DelayStatesParams(DataAdapterParamsBase):
    active_joints: list[int]
    dt: float
    ctrl_step: int
    delay_step: int
    include_dqdes: bool = False


class DelayStates(DataAdapterBase[DelayStatesParams, DefaultStates, DefaultRefs, DefaultInputs]):
    states_queue: deque[np.ndarray]

    def __init__(self, params: DelayStatesParams) -> None:
        super().__init__(params)
        if params.ctrl_step < 1:
            msg = f"DelayStatesParams.ctrl_step must be larger than or equal to 1: {params.ctrl_step}"
            raise ValueError(msg)
        if params.delay_step < 0:
            msg = f"DelayStatesParams.delay_step must be larger than or equal to 0: {params.delay_step}"
            raise ValueError(msg)
        self.reset()

    def make_feature(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        delay_step = self.params.delay_step
        shift = ctrl_step + delay_step

        states_keys = _get_keys(["q", "dq", "pa", "pb"], joints)
        refs: list[str] = ["q"]
        if self.params.include_dqdes:
            refs.append("dq")
        ref_keys = _get_keys(refs, joints)

        delayed_states = extract_data(dataset, states_keys, end=-shift)
        current_states = extract_data(dataset, states_keys, start=delay_step, end=-ctrl_step)
        reference = extract_data(
            dataset,
            ref_keys,
            start=shift,
            keys_replace=_get_keys([f"{x}des" for x in refs], joints),
        )
        feature_data = pd.concat((delayed_states, current_states, reference), axis=1)
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        delay_step = self.params.delay_step
        ctrl_input = extract_data(dataset, _get_keys(["ca", "cb"], joints), start=delay_step, end=-ctrl_step)
        return ctrl_input.to_numpy()

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        joints = self.params.active_joints
        q = states["q"][joints]
        dq = states["dq"][joints]
        pa = states["pa"][joints]
        pb = states["pb"][joints]
        current_states = np.concatenate((q, dq, pa, pb))
        self.states_queue.append(current_states)
        delayed_states = self.states_queue.popleft()

        qdes = refs["qdes"](t)[joints]
        if self.params.include_dqdes:
            dqdes = refs["dqdes"](t)[joints]
            model_input = np.concatenate((delayed_states, current_states, qdes, dqdes))
        else:
            model_input = np.concatenate((delayed_states, current_states, qdes))
        return np.atleast_2d(model_input)

    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        # `y` has has 1 row and n_targets columns.
        joints = self.params.active_joints
        y = np.ravel(y)
        ca, cb = base_inputs["ca"], base_inputs["cb"]
        n = len(joints)
        ca[joints] = y[:n]
        cb[joints] = y[n:]
        return ca, cb

    def reset(self) -> None:
        n_states = len(self.params.active_joints) * len(["q", "dq", "pa", "pb"])
        self.states_queue = deque([np.zeros(shape=n_states, dtype=float)] * self.params.delay_step)


@dataclass
class DelayStatesAllParams(DataAdapterParamsBase):
    active_joints: list[int]
    dt: float
    ctrl_step: int
    delay_step: int
    include_dqdes: bool = False


class DelayStatesAll(DataAdapterBase[DelayStatesAllParams, DefaultStates, DefaultRefs, DefaultInputs]):
    states_queue: list[np.ndarray]

    def __init__(self, params: DelayStatesAllParams) -> None:
        super().__init__(params)
        if params.ctrl_step < 1:
            msg = f"DelayStatesParams.ctrl_step must be larger than or equal to 1: {params.ctrl_step}"
            raise ValueError(msg)
        if params.delay_step < 0:
            msg = f"DelayStatesParams.delay_step must be larger than or equal to 0: {params.delay_step}"
            raise ValueError(msg)
        self.reset()

    def make_feature(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        delay_step = self.params.delay_step
        shift = ctrl_step + delay_step

        states_keys = _get_keys(["q", "dq", "pa", "pb"], joints)
        refs: list[str] = ["q"]
        if self.params.include_dqdes:
            refs.append("dq")
        ref_keys = _get_keys(refs, joints)

        states = extract_data(dataset, states_keys, start=delay_step, end=-ctrl_step)
        for i in range(1, delay_step + 1):
            start = delay_step - i
            end = ctrl_step + i
            delayed_states = extract_data(dataset, states_keys, start=start, end=-end)
            states = pd.concat((delayed_states, states), axis=1)
        reference = extract_data(
            dataset,
            ref_keys,
            start=shift,
            keys_replace=_get_keys([f"{x}des" for x in refs], joints),
        )
        feature_data = pd.concat((states, reference), axis=1)
        return feature_data.to_numpy()

    def make_target(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        ctrl_step = self.params.ctrl_step
        delay_step = self.params.delay_step
        ctrl_input = extract_data(dataset, _get_keys(["ca", "cb"], joints), start=delay_step, end=-ctrl_step)
        return ctrl_input.to_numpy()

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        joints = self.params.active_joints
        q = states["q"][joints]
        dq = states["dq"][joints]
        pa = states["pa"][joints]
        pb = states["pb"][joints]
        current_states = np.concatenate((q, dq, pa, pb))
        self.states_queue.append(current_states)
        concat_states = np.ravel(self.states_queue)
        self.states_queue.pop(0)

        qdes = refs["qdes"](t)[joints]
        if self.params.include_dqdes:
            dqdes = refs["dqdes"](t)[joints]
            model_input = np.concatenate((concat_states, qdes, dqdes))
        else:
            model_input = np.concatenate((concat_states, qdes))
        return np.atleast_2d(model_input)

    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        # `y` has has 1 row and n_targets columns.
        joints = self.params.active_joints
        y = np.ravel(y)
        ca, cb = base_inputs["ca"], base_inputs["cb"]
        n = len(joints)
        ca[joints] = y[:n]
        cb[joints] = y[n:]
        return ca, cb

    def reset(self) -> None:
        n_states = len(self.params.active_joints) * len(["q", "dq", "pa", "pb"])
        self.states_queue = [np.zeros(shape=n_states, dtype=float)] * self.params.delay_step


class Scaler(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Scaler: ...  # noqa: N803

    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...  # noqa: N803

    def transform(self, X: np.ndarray) -> np.ndarray | Unknown: ...  # noqa: N803

    def get_params(self) -> dict[str, object]: ...


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Regressor: ...  # noqa: N803

    def predict(self, X: np.ndarray) -> np.ndarray | Unknown: ...  # noqa: N803

    def score(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float | Unknown: ...

    def get_params(self) -> dict[str, object]: ...


def load_model_config_file(config_file: str | Path) -> dict[str, Unknown]:
    with Path(config_file).open("rb") as f:
        return tomllib.load(f)


DATA_ADAPTER_MAP: Mapping[str, tuple[type[DataAdapterBase], type[DataAdapterParamsBase]]] = {
    "preview-ref": (PreviewRef, PreviewRefParams),
    "delay-states": (DelayStates, DelayStatesParams),
    "delay-states-all": (DelayStatesAll, DelayStatesAllParams),
    #"multi-horizon": (MultiHorizonHandler, MultiHorizonParams),
    "single-shot-multi-horizon": (SingleShotMultiHorizonHandler, SingleShotMultiHorizonParams),
}


SCALER_MAP: Mapping[str, tuple[type[Scaler], None]] = {
    "std": (StandardScaler, None),
    "minmax": (MinMaxScaler, None),
    "maxabs": (MaxAbsScaler, None),
    "robust": (RobustScaler, None),
}


REGRESSOR_MAP: Mapping[str, tuple[type[Regressor], None]] = {
    "linear": (LinearRegression, None),
    "ridge": (Ridge, None),
    "mlp": (MLPRegressor, None),
}


def pop_multi_keys(config: dict[str, Unknown], keys: Iterable[str]) -> dict[str, Unknown]:
    return {key: config.pop(key, {}) for key in keys}


def _tweak_config_values(config: dict[str, Unknown]) -> dict[str, Unknown]:
    for key, value in config.items():
        if isinstance(value, str) and value.lower() == "none":
            config[key] = None

        match key:
            case "feature_range":
                config[key] = tuple(value)  # MinMaxScaler
            case "quantile_range":
                config[key] = tuple(value)  # RobustScaler
            case "hidden_layer_sizes":
                config[key] = tuple(value)  # MLPRegressor
    return config


def _load_from_map(
    config: dict[str, Unknown],
    _map: Mapping[str, tuple[type[T], type[DataAdapterParamsBase] | None]],
    _display: str,
) -> T:
    try:
        name = config.pop("name")
    except KeyError as e:
        msg = f"'name' is required to load {_display}"
        raise KeyError(msg) from e

    try:
        _type, params_type = _map[name]
    except KeyError as e:
        msg = f"unknown {_display} name: {name}"
        raise KeyError(msg) from e

    params_set = pop_multi_keys(config, _map.keys())
    selected_params_set = config.pop("params", None)
    if isinstance(selected_params_set, str):
        try:
            config.update(params_set[name][selected_params_set])
        except KeyError as e:
            msg = f"unknown parameter set name: {selected_params_set}"
            raise KeyError(msg) from e
    elif selected_params_set is not None:
        msg = f"value of 'params' is not string: {selected_params_set}"
        raise ValueError(msg)

    _tweak_config_values(config)
    if params_type is None:
        return _type(**config)
    return _type(params_type(**config))  # type: ignore[call-arg]


def update_config_by_selector(
    config: dict[str, Unknown],
    selector: str,
) -> dict[str, Unknown]:
    splitted = selector.split(".")
    config.update(name=splitted[0])
    if len(splitted) > 1:
        config.update(params=splitted[1])
    return config


def load_data_adapter(
    config: dict[str, Unknown],
    active_joints: list[int] | None = None,
    selector: str | None = None,
) -> DataAdapterBase:
    if selector is not None:
        config = update_config_by_selector(config, selector)
    adapter = _load_from_map(config, DATA_ADAPTER_MAP, "data adapter")
    if active_joints is not None:
        adapter.params.active_joints = active_joints
    event_logger().debug("Loaded data adapter: %s", adapter)
    return adapter


def load_scaler(config: dict[str, Unknown], selector: str | None = None) -> Scaler | None:
    if selector is not None:
        config = update_config_by_selector(config, selector)
    scaler = None
    if config.get("name", "x").lower() != "none":
        scaler = _load_from_map(config, SCALER_MAP, "scaler")
    event_logger().debug("Loaded scaler: %s", scaler)
    return scaler


def load_regressor(config: dict[str, Unknown], selector: str | None = None) -> Regressor:
    if selector is not None:
        config = update_config_by_selector(config, selector)
    regressor = _load_from_map(config, REGRESSOR_MAP, "regressor")
    event_logger().debug("Loaded regressor: %s", regressor)
    return regressor


def load_model(
    config: dict[str, Unknown],
    scaler_selector: str | None = None,
    regressor_selector: str | None = None,
) -> Regressor | Pipeline:
    scaler = load_scaler(config["scaler"], scaler_selector)
    regressor = load_regressor(config["regressor"], regressor_selector)
    if scaler is None:
        return regressor
    return make_pipeline(scaler, regressor)


def extract_data(
    dataset: Data,
    keys: Iterable[str],
    *,
    start: int | None = None,
    end: int | None = None,
    keys_replace: Iterable[str] | None = None,
) -> pd.DataFrame:
    subset = dataset.df.loc[:, keys]
    if start is None:
        start = 0
    if end is None:
        end = len(subset)
    extracted = subset[start:end]
    if keys_replace is not None:
        extracted = extracted.rename(columns=dict(zip(keys, keys_replace, strict=True)))
    return extracted.reset_index(drop=True)


def _load_datasets(
    data_file_or_directory: str | Path,
    glob_pattern: str,
    n_pickup: int | None,
) -> list[Data]:
    collection: list[Data] = []
    path = Path(data_file_or_directory)
    if path.is_dir():
        collection.extend(Data(x) for x in sorted(path.glob(glob_pattern)))
        if n_pickup is not None:
            collection = collection[:n_pickup]
        event_logger().debug("%s datasets loaded from %s", len(collection), path)
    else:
        collection.append(Data(path))

    return collection


def load_datasets(
    dataset_paths: str | Path | Iterable[str | Path],
    glob_pattern: str = "**/*.csv",
    n_pickup: int | None = None,
) -> list[Data]:
    if isinstance(dataset_paths, str | Path):
        dataset_paths = [dataset_paths]

    event_logger().debug("Loading datasets from %s", list(map(str, dataset_paths)))
    datasets: list[Data] = []
    for dataset_path in dataset_paths:
        datasets.extend(_load_datasets(dataset_path, glob_pattern, n_pickup))
    event_logger().info("%s datasets loaded in total", len(datasets))

    if len(datasets) == 0:
        msg = f"No dataset found with {glob_pattern}: {dataset_paths}"
        raise RuntimeError(msg)
    return datasets


def load_train_datasets(
    train_datasets: Data | Iterable[Data],
    adapter: DataAdapterBase[DataAdapterParamsType, StatesType, RefsType, InputsType],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(train_datasets, Data):
        train_datasets = [train_datasets]
    x_train = cast(np.ndarray, None)
    y_train = cast(np.ndarray, None)
    for dataset in train_datasets:
        _x_train = adapter.make_feature(dataset)
        _y_train = adapter.make_target(dataset)
        x_train = np.vstack((x_train, _x_train)) if x_train is not None else np.copy(_x_train)
        y_train = np.vstack((y_train, _y_train)) if y_train is not None else np.copy(_y_train)
        if dataset.is_loaded_from_file():
            event_logger().debug("Loaded dataset: %s", dataset.datapath)
    if x_train is None or y_train is None:
        msg = f"No data sets found: {train_datasets}"
        raise RuntimeError(msg)
    if len(y_train.shape) == 2 and y_train.shape[1] == 1:  # noqa: PLR2004
        y_train = np.ravel(y_train)
    return x_train, y_train


@dataclass
class TrainedModel(Generic[DataAdapterParamsType, StatesType, RefsType, InputsType]):
    model: Regressor | Pipeline
    adapter: DataAdapterBase[DataAdapterParamsType, StatesType, RefsType, InputsType]
    initial_tau: int | None = None

    def get_params(self) -> dict:
        return self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray | Unknown:  # noqa: N803
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float | Unknown:  # noqa: N803
        return self.model.score(X, y, sample_weight)

    @property
    def model_name(self) -> str:
        if isinstance(self.model, Pipeline):
            return " -> ".join(str(step[1]) for step in self.model.steps)
        return str(self.model)


def train_model(
    model: Regressor | Pipeline,
    datasets: Data | Iterable[Data],
    adapter: DataAdapterBase[...],
    val_size: float = 0.25, # 検証データの割合を追加
    seed: int | None = None,
) -> TrainedModel:
    # データセットを学習用と検証用に分割
    from sklearn.model_selection import train_test_split
    if isinstance(datasets, Data):
        datasets = [datasets]
    train_datasets, val_datasets = train_test_split(list(datasets), test_size=val_size, random_state=seed)

    # 学習データでモデルを訓練
    x_train, y_train = load_train_datasets(train_datasets, adapter)
    event_logger().debug("x_train.shape = %s", x_train.shape)
    event_logger().debug("y_train.shape = %s", y_train.shape)
    model = model.fit(x_train, y_train)

    # ★ ここからが追加ロジック ★
    # 検証データで最適な初期tauを見つける
    initial_tau = None
    if val_datasets:
        event_logger().info("Finding best initial tau from validation set...")
        x_val, y_val = load_train_datasets(val_datasets, adapter)
        y_pred = model.predict(x_val)
        
        best_score = -np.inf
        
        # モデルの出力（長いベクトル）と正解ラベルを各tauごとに分割して比較
        N_min = adapter.params.min_preview_step
        N_max = adapter.params.max_preview_step
        n_ctrl_features = y_val.shape[1] // (N_max - N_min + 1)

        for k in range(N_min, N_max + 1):
            start_idx = (k - N_min) * n_ctrl_features
            end_idx = start_idx + n_ctrl_features
            
            y_true_k = y_val[:, start_idx:end_idx]
            y_pred_k = y_pred[:, start_idx:end_idx]
            
            score_k = r2_score(y_true_k, y_pred_k)
            event_logger().info(f"  - R^2 score for tau={k}: {score_k:.4f}")
            if score_k > best_score:
                best_score = score_k
                initial_tau = k
        event_logger().info(f"Best initial tau found: {initial_tau} (score: {best_score:.4f})")

    # 見つけたinitial_tauをモデルと一緒に保存
    return TrainedModel(model, adapter, initial_tau=initial_tau)


def dump_trained_model(trained_model: TrainedModel, output: str | Path) -> Path:
    joblib.dump(trained_model, output)
    return Path(output)


def load_trained_model(model_filepath: str | Path) -> TrainedModel:
    return joblib.load(model_filepath)


CtrlAdapterUpdater: TypeAlias = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, RefFuncType, RefFuncType],
    tuple[np.ndarray, np.ndarray],
]


class CtrlAdapter(Generic[DataAdapterParamsType, StatesType, RefsType, InputsType]):
    ctrl: AffPosCtrl
    model: TrainedModel[DataAdapterParamsType, StatesType, RefsType, InputsType]
    _updater: CtrlAdapterUpdater
    preview_steps: int
    warmup_steps: int
    _n_steps: int
    _preview_time: float

    def __init__(
        self,
        ctrl: AffPosCtrl,
        model: TrainedModel[DataAdapterParamsType, StatesType, RefsType, InputsType] | None,
        preview_steps: int,
        warmup_steps: int,
    ) -> None:
        self.ctrl = ctrl
        if model is None:
            self._updater = self.update_ctrl
        else:
            self.model = model
            self._updater = self.update_model
        self.preview_steps = preview_steps
        self.warmup_steps = warmup_steps
        self._n_steps = 0
        self._preview_time = self.preview_steps * self.ctrl.dt

    def update_ctrl(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: RefFuncType,
        dqdes: RefFuncType,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_ref = t + self._preview_time
        return self.ctrl.update(t, q, dq, pa, pb, qdes(t_ref), dqdes(t_ref))

    def update_model(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: RefFuncType,
        dqdes: RefFuncType,
    ) -> tuple[np.ndarray, np.ndarray]:
        ca, cb = self.ctrl.update(t, q, dq, pa, pb, qdes(t), dqdes(t))
        return ca, cb

    def update(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: RefFuncType,
        dqdes: RefFuncType,
    ) -> tuple[np.ndarray, np.ndarray]:
        ca, cb = self._updater(t, q, dq, pa, pb, qdes, dqdes)
        self._n_steps += 1
        return ca, cb


class DefaultCtrlAdapter(CtrlAdapter[DataAdapterParamsType, DefaultStates, DefaultRefs, DefaultInputs]):
    def __init__(
        self,
        ctrl: AffPosCtrl,
        model: TrainedModel[DataAdapterParamsType, DefaultStates, DefaultRefs, DefaultInputs] | None,
        preview_steps: int,
        warmup_steps: int,
    ) -> None:
        super().__init__(ctrl, model, preview_steps, warmup_steps)

    def update_model(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: RefFuncType,
        dqdes: RefFuncType,
    ) -> tuple[np.ndarray, np.ndarray]:
        ca, cb = self.ctrl.update(t, q, dq, pa, pb, qdes(t), dqdes(t))
        x = self.model.adapter.make_model_input(
            t,
            {"q": q, "dq": dq, "pa": pa, "pb": pb},
            {"qdes": qdes, "dqdes": dqdes},
        )
        y = self.model.predict(x)
        if self._n_steps < self.warmup_steps:
            return ca, cb
        ca, cb = self.model.adapter.make_ctrl_input(y, {"ca": ca, "cb": cb})
        return ca, cb


DefaultCtrlAdapterType: TypeAlias = DefaultCtrlAdapter[DataAdapterParamsBase]
DefaultTrainedModelType: TypeAlias = TrainedModel[DataAdapterParamsBase, DefaultStates, DefaultRefs, DefaultInputs]


def control_position_or_model(
    controller: CONTROLLER_T,
    model: DefaultTrainedModelType | None,
    qdes_func: RefFuncType,
    dqdes_func: RefFuncType,
    duration: float,
    preview_steps: int,
    logger: Logger | None = None,
    log_filename: str | Path | None = None,
    time_updater: str = "accumulated",
    header_text: str = "",
    warmup_steps: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    reset_logger(logger, log_filename)
    comm, ctrl, state = controller
    ctrl_adapter = DefaultCtrlAdapter(ctrl, model, preview_steps, warmup_steps)
    if model is not None:
        model.adapter.reset()
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
        ca, cb = ctrl_adapter.update(t, q, dq, pa, pb, qdes_func, dqdes_func)
        comm.send_commands(ca, cb)
        if logger is not None:
            logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes_func(t), dqdes_func(t))
        timer.block()
    sys.stdout.write("\n")
    # Return the last commands that have been sent to the valve.
    return ca, cb

@dataclass
class MultiHorizonParams(DataAdapterParamsBase):
    """
    MultiHorizonHandlerが使用するパラメータ。
    """
    active_joints: list[int]
    dt: float
    ctrl_step: int
    min_preview_step: int
    max_preview_step: int

class MultiHorizonHandler(DataAdapterBase[MultiHorizonParams, DefaultStates, DefaultRefs, DefaultInputs]):
    """
    適応制御モデルのためのマルチホライズン学習データセットを生成するハンドラ。
    """
    def __init__(self, params: MultiHorizonParams) -> None:
        super().__init__(params)
        if params.max_preview_step < 1:
            msg = f"max_preview_step must be larger than or equal to 1: {params.max_preview_step}"
            raise ValueError(msg)
        if params.min_preview_step < 1 or params.min_preview_step > params.max_preview_step:
            msg = f"min_preview_step is out of valid range: {params.min_preview_step}"
            raise ValueError(msg)
        self.reset()
        self._cached_X: np.ndarray | None = None
        self._cached_y: np.ndarray | None = None

    def _make_dataset(self, dataset: Data) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_X is not None and self._cached_y is not None:
            return self._cached_X, self._cached_y

        X_list = []
        y_list = []

        joints = self.params.active_joints
        N_min = self.params.min_preview_step
        N_max = self.params.max_preview_step
        
        state_keys = _get_keys(["q", "dq", "pa", "pb"], joints)
        ref_keys = _get_keys(["q"], joints)
        ctrl_keys = _get_keys(["ca", "cb"], joints)

        df_state = dataset.df.loc[:, state_keys]
        df_ref = dataset.df.loc[:, ref_keys]
        df_ctrl = dataset.df.loc[:, ctrl_keys]
        
        n_ref_features = len(ref_keys)
        valid_length = len(df_state) - N_max

        for t in range(valid_length):
            for k in range(N_min, N_max + 1):
                s_t = df_state.iloc[t].values
                e_k = np.zeros(N_max)
                e_k[k - 1] = 1.0
                r_masked_flat = np.zeros(N_max * n_ref_features)
                r_t_plus_k = df_ref.iloc[t + k].values
                start_idx = (k - 1) * n_ref_features
                end_idx = k * n_ref_features
                r_masked_flat[start_idx:end_idx] = r_t_plus_k
                x_sample = np.concatenate([s_t, r_masked_flat, e_k])
                X_list.append(x_sample)
                y_sample = df_ctrl.iloc[t + k].values
                y_list.append(y_sample)
        
        self._cached_X = np.array(X_list)
        self._cached_y = np.array(y_list)
        return self._cached_X, self._cached_y

    def make_feature(self, dataset: Data) -> np.ndarray:
        X, _ = self._make_dataset(dataset)
        return X

    def make_target(self, dataset: Data) -> np.ndarray:
        _, y = self._make_dataset(dataset)
        return y
    
    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        raise NotImplementedError("Online logic for MultiHorizon is not implemented yet.")

    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        raise NotImplementedError("Online logic for MultiHorizon is not implemented yet.")

    def reset(self) -> None:
        self._cached_X = None
        self._cached_y = None


# Local Variables:
# jinx-local-words: "MLPRegressor Params apdater arg cb csv ctrl dataset datasets des dq dqdes maxabs minmax mlp noqa npqa params pb qdes quantile rb regressor scaler" # noqa: E501
# End:
