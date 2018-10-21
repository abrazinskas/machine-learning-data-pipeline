from mldp.utils.util_classes.fs_accessors.local_fs_accessor import\
    LocalFsAccessor
from mldp.utils.util_classes.fs_accessors.s3_fs_accessor import S3FsAccessor


def fs_accessor_factory(fs_type="local"):
    assert fs_type in ["local", "s3"]
    if fs_type == "local":
        return LocalFsAccessor()
    if fs_type == "s3":
        return S3FsAccessor()
