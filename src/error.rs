//! Error type for the fallible operations of this crate.

use std::fmt;

/// Errors reported by operations whose preconditions depend on run-time state, such as refitting a
/// tree that was built with spatial splits or loading a file that was written by another layout.
///
/// Programmer errors that are always detectable from the arguments alone (empty geometry, an index
/// out of range, a vertex count that is not a multiple of three) panic instead, in the same way that
/// slice indexing does.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The structure has not been built yet, or the build produced no nodes.
    NotBuilt,
    /// The tree cannot be refitted: it was built with spatial splits or presplitting, it may contain
    /// unused nodes, or it is a TLAS (rebuild a TLAS instead).
    NotRefittable,
    /// The layout has no intermediate tree to refit or optimize in place: it was converted from
    /// another structure, loaded from a file, or built with `discard_intermediates`. Build it from
    /// geometry instead.
    NotRebuildable,
    /// The operation is only available on a top-level acceleration structure.
    NotTlas,
    /// The operation needs triangle geometry, but the structure was built over custom primitives,
    /// boxes or instances, or its geometry was released.
    NoGeometry,
    /// The tree may contain unused nodes (after a threaded build or leaf merging); compact it first.
    HasHoles,
    /// `build_simd` was called, but tinybvh was compiled without a SIMD builder for this target.
    SimdBuilderUnavailable,
    /// The file could not be loaded: it does not exist, was written by a different tinybvh version
    /// or layout, or does not match the geometry it is bound to.
    LoadFailed(String),
    /// The path contains an interior NUL byte and cannot be passed to the C library.
    InvalidPath,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotBuilt => write!(f, "the acceleration structure has not been built"),
            Error::NotRefittable => write!(
                f,
                "the tree cannot be refitted (spatial splits, presplitting, unused nodes, or a TLAS)"
            ),
            Error::NotRebuildable => write!(
                f,
                "the layout has no intermediate tree (converted, loaded, or intermediates discarded) and cannot be refitted or optimized"
            ),
            Error::NotTlas => write!(f, "the operation requires a top-level acceleration structure"),
            Error::NoGeometry => write!(f, "the operation requires triangle geometry"),
            Error::HasHoles => write!(f, "the tree may contain unused nodes; compact it first"),
            Error::SimdBuilderUnavailable => {
                write!(f, "tinybvh was compiled without a SIMD builder for this target")
            }
            Error::LoadFailed(path) => write!(f, "could not load acceleration structure from `{path}`"),
            Error::InvalidPath => write!(f, "the path contains an interior NUL byte"),
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias for `Result<T, tinybvh::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
