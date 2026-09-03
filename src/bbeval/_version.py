"""Single source of the package version.

The version is part of every run's config fingerprint, so a release that
changes scoring behaviour invalidates earlier artefacts instead of silently
resuming on top of them. Bump it whenever a change alters results.
"""

__version__ = "0.3.0"
