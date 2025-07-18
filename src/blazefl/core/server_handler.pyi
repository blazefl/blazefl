from typing import Protocol, TypeVar

UplinkPackage = TypeVar('UplinkPackage')
DownlinkPackage = TypeVar('DownlinkPackage', covariant=True)

class BaseServerHandler(Protocol[UplinkPackage, DownlinkPackage]):
    def downlink_package(self) -> DownlinkPackage: ...
    def sample_clients(self) -> list[int]: ...
    def if_stop(self) -> bool: ...
    def global_update(self, buffer: list[UplinkPackage]) -> None: ...
    def load(self, payload: UplinkPackage) -> bool: ...
