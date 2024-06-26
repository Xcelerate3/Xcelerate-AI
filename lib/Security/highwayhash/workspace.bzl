"""loads the highwayhash library, used by XC."""

load("//third_party:repo.bzl", "XC_http_archive", "XC_mirror_urls")

def repo():
    XC_http_archive(
        name = "highwayhash",
        urls = XC_mirror_urls("https://github.com/google/highwayhash/archive/c13d28517a4db259d738ea4886b1f00352a3cc33.tar.gz"),
        sha256 = "c0e2b9931fbcce3bfbcd7999c3c114f404ac0f8b89775a5bbccbcaa501868e58",
        strip_prefix = "highwayhash-c13d28517a4db259d738ea4886b1f00352a3cc33",
        build_file = "//third_party/highwayhash:highwayhash.BUILD",
    )
