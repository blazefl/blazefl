import signal
import threading
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.pool import ApplyResult
from pathlib import Path
from typing import Literal, Protocol, TypeVar

import torch
import torch.multiprocessing as mp

from blazefl.core.utils import process_tensors_in_object, reconstruct_from_shared_memory

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage", contravariant=True)


class BaseClientTrainer(Protocol[UplinkPackage, DownlinkPackage]):
    """
    Abstract base class for serial client training in federated learning.

    This class defines the interface for training clients in a serial manner,
    where each client is processed one after the other.

    Raises:
        NotImplementedError: If the methods are not implemented in a subclass.
    """

    def uplink_package(self) -> list[UplinkPackage]:
        """
        Prepare the data package to be sent from the client to the server.

        Returns:
            list[UplinkPackage]: A list of data packages prepared for uplink
            transmission.
        """
        ...

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        """
        Process the downlink payload from the server for a list of client IDs.

        Args:
            payload (DownlinkPackage): The data package received from the server.
            cid_list (list[int]): A list of client IDs to process.

        Returns:
            None
        """
        ...


ClientConfig = TypeVar("ClientConfig")


class ProcessPoolClientTrainer(
    BaseClientTrainer[UplinkPackage, DownlinkPackage],
    Protocol[UplinkPackage, DownlinkPackage, ClientConfig],
):
    """
    Abstract base class for parallel client training in federated learning.

    This class extends SerialClientTrainer to enable parallel processing of clients,
    allowing multiple clients to be trained concurrently.

    Attributes:
        num_parallels (int): Number of parallel processes to use for client training.
        share_dir (Path): Directory path for sharing data between processes.
        device (str): The primary device to use for computation (e.g., "cpu", "cuda").
        device_count (int): The number of available CUDA devices, if `device` is "cuda".
        cache (list[UplinkPackage]): Cache to store uplink packages from clients.
        ipc_mode (Literal["storage", "shared_memory"]): Inter-process communication
            mode. "storage" uses disk for data exchange, "shared_memory" uses
            shared memory for tensor data. Defaults to "storage".

    Raises:
        NotImplementedError: If the abstract methods are not implemented in a subclass.
    """

    num_parallels: int
    share_dir: Path
    device: str
    device_count: int
    cache: list[UplinkPackage]
    ipc_mode: Literal["storage", "shared_memory"] = "storage"
    stop_event: threading.Event

    def progress_fn(
        self,
        it: list[ApplyResult],
    ) -> Iterable[ApplyResult]:
        """
        A no-op progress function that can be overridden to provide custom
        progress tracking.

        Args:
            it (list[ApplyResult]): A list of ApplyResult objects.

        Returns:
            Iterable[ApplyResult]: The original iterable.
        """
        return it

    def get_client_config(self, cid: int) -> ClientConfig:
        """
        Retrieve the configuration for a given client ID.

        Args:
            cid (int): Client ID.

        Returns:
            ClientConfig: The configuration for the specified client.
        """
        ...

    def get_client_device(self, cid: int) -> str:
        """
        Retrieve the device to use for processing a given client.

        Args:
            cid (int): Client ID.

        Returns:
            str: The device to use for processing the client.
        """
        if self.device == "cuda":
            return f"cuda:{cid % self.device_count}"
        return self.device

    @staticmethod
    def worker(
        config: ClientConfig | Path,
        payload: DownlinkPackage | Path,
        device: str,
        stop_event: threading.Event,
        *,
        shm_buffer: UplinkPackage | None = None,
    ) -> UplinkPackage | Path:
        """
        Process a single client's training task.

        This method is executed by each worker process in the pool.
        It handles loading client configuration and payload, performing
        the client-specific operations, and returning the result.

        Args:
            config (ClientConfig | Path):
                The client's configuration data, or a path to a file containing
                the configuration if `ipc_mode` is "storage".
            payload (DownlinkPackage | Path):
                The downlink payload from the server, or a path to a file
                containing the payload if `ipc_mode` is "storage".
            device (str): Device to use for processing (e.g., "cpu", "cuda:0").

        Returns:
            UplinkPackage | Path:
                The uplink package containing the client's results, or a path
                to a file containing the package if `ipc_mode` is "storage".
        """
        ...

    def prepare_uplink_package_buffer(self) -> UplinkPackage:
        raise NotImplementedError

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        """
        Manage the parallel processing of clients.

        This method distributes the processing of multiple clients across
        parallel processes, handling data saving, loading, and caching.

        Args:
            payload (DownlinkPackage): The data package received from the server.
            cid_list (list[int]): A list of client IDs to process.

        Returns:
            None
        """
        payload_path = Path()
        shm_buffers = {}
        if self.ipc_mode == "storage":
            payload_path = self.share_dir.joinpath("payload.pkl")
            torch.save(payload, payload_path)
        else:  # shared_memory
            process_tensors_in_object(payload, mode="move")
            for cid in cid_list:
                buffer = self.prepare_uplink_package_buffer()
                process_tensors_in_object(buffer, mode="move")
                shm_buffers[cid] = buffer

        self.stop_event.clear()
        pool = mp.Pool(
            processes=self.num_parallels,
            initializer=signal.signal,
            initargs=(signal.SIGINT, signal.SIG_IGN),
        )
        try:
            jobs: list[ApplyResult] = []
            for cid in cid_list:
                config = self.get_client_config(cid)
                device = self.get_client_device(cid)
                if self.ipc_mode == "storage":
                    config_path = self.share_dir.joinpath(f"{cid}.pkl")
                    torch.save(config, config_path)
                    jobs.append(
                        pool.apply_async(
                            self.worker,
                            (config_path, payload_path, device, self.stop_event),
                        ),
                    )
                else:  # shared_memory
                    jobs.append(
                        pool.apply_async(
                            self.worker,
                            (
                                config,
                                payload,
                                device,
                                self.stop_event,
                            ),
                            kwds={
                                "shm_buffer": shm_buffers.get(cid),
                            },
                        ),
                        # )
                    )

            for i, job in enumerate(self.progress_fn(jobs)):
                result = job.get()
                if self.ipc_mode == "storage":
                    assert isinstance(result, Path)
                    package = torch.load(result, weights_only=False)
                else:  # shared_memory
                    cid = cid_list[i]
                    package = reconstruct_from_shared_memory(result, shm_buffers[cid])
                self.cache.append(package)
        finally:
            self.stop_event.set()
            pool.close()
            pool.join()


class ThreadPoolClientTrainer(
    BaseClientTrainer[UplinkPackage, DownlinkPackage],
    Protocol[UplinkPackage, DownlinkPackage],
):
    num_parallels: int
    device: str
    device_count: int
    cache: list[UplinkPackage]
    stop_event: threading.Event

    def progress_fn(
        self, it: list[Future[UplinkPackage]]
    ) -> Iterable[Future[UplinkPackage]]:
        """
        A no-op progress function that can be overridden to provide custom
        progress tracking.

        Args:
            it (list[Future[UplinkPackage]]): A list of Future objects
                representing the results of client processing.

        Returns:
            Iterable[Future[UplinkPackage]]: The original iterable.
        """
        return it

    def worker(
        self,
        cid: int,
        device: str,
        payload: DownlinkPackage,
        stop_event: threading.Event,
    ) -> UplinkPackage:
        """
        Process a single client's training task in a thread.

        Args:
            cid (int): The client ID.
            device (str): The device to use for processing this client.
            payload (DownlinkPackage): The data package received from the server.
            stop_event (threading.Event): Event to signal stopping the worker.

        Returns:
            UplinkPackage: The uplink package containing the client's results.
        """
        ...

    def get_client_device(self, cid: int) -> str:
        if self.device == "cuda":
            return f"cuda:{cid % self.device_count}"
        return self.device

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        self.stop_event.clear()
        executor = ThreadPoolExecutor(max_workers=self.num_parallels)
        try:
            futures: list[Future[UplinkPackage]] = []
            for cid in cid_list:
                device = self.get_client_device(cid)
                future = executor.submit(
                    self.worker,
                    cid,
                    device,
                    payload,
                    self.stop_event,
                )
                futures.append(future)

            for future in self.progress_fn(futures):
                result = future.result()
                self.cache.append(result)
        finally:
            self.stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
