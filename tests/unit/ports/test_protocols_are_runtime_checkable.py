from pathlib import Path

from readeverything.adapters.local_source import LocalFileSource
from readeverything.ports.artifacts import ArtifactStore
from readeverything.ports.source import FileSource, SourceLister, SourceReader, SourceStat


def test_a_real_adapter_satisfies_its_port_structurally(tmp_path: Path) -> None:
    """Structural typing is the point: an adapter must not have to inherit."""
    source = LocalFileSource(root=tmp_path)
    assert isinstance(source, SourceStat)
    assert isinstance(source, SourceReader)
    assert isinstance(source, SourceLister)
    assert isinstance(source, FileSource)


def test_an_unrelated_object_does_not_satisfy_a_port() -> None:
    """...and the check must be able to say no."""
    assert not isinstance(object(), FileSource)
    assert not isinstance(object(), ArtifactStore)


def test_file_source_composes_the_narrow_slices() -> None:
    """Collaborators annotate the slimmest slice they use."""
    assert issubclass(FileSource, SourceStat)
    assert issubclass(FileSource, SourceReader)
    assert issubclass(FileSource, SourceLister)
