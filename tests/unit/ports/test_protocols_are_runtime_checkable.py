from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.detection import MimeDetector
from readeverything.ports.handler import MediaHandler
from readeverything.ports.source import FileSource, SourceLister, SourceReader, SourceStat

PORTS = [
    ArtifactStore,
    MimeDetector,
    MediaHandler,
    FileSource,
    SourceLister,
    SourceReader,
    SourceStat,
]


def test_every_port_is_runtime_checkable() -> None:
    """Structural typing is the point: an adapter must not have to inherit."""
    for port in PORTS:
        assert hasattr(port, "_is_runtime_protocol"), f"{port.__name__} is not runtime_checkable"


def test_file_source_composes_the_narrow_slices() -> None:
    """Collaborators annotate the slimmest slice they use."""
    assert issubclass(FileSource, SourceStat)
    assert issubclass(FileSource, SourceReader)
    assert issubclass(FileSource, SourceLister)
