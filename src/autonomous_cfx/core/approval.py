"""Autonomous CFX Studio - Independent Human vs Synthetic Test Approval Management.
Enforces strict separation between human artist sign-off and test harness execution.
Synthetic test events must never authorize production delivery outside explicitly isolated test mode.
"""

from __future__ import annotations

import getpass
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ApprovalAuthority(str, Enum):
    HUMAN_ARTIST = "human_artist"
    TEST_HARNESS_SYNTHETIC = "test_harness_synthetic"


@dataclass
class ApprovalRecord:
    """Cryptographically verifiable record of simulation approval."""
    authority: ApprovalAuthority
    actor_id: str
    approved_version: int
    cache_hash: str
    timestamp: float = field(default_factory=time.time)
    review_notes: str = ""
    is_production_authorized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority": self.authority.value,
            "actor_id": self.actor_id,
            "approved_version": self.approved_version,
            "cache_hash": self.cache_hash,
            "timestamp": self.timestamp,
            "review_notes": self.review_notes,
            "is_production_authorized": self.is_production_authorized,
        }

    @staticmethod
    def compute_cache_hash(
        frames: List[MeshGeometry],
        version: int,
        shot_name: str = "shot020"
    ) -> str:
        """Compute cryptographic SHA-256 digest of cache vertex content bound to version and shot."""
        hasher = hashlib.sha256()
        hasher.update(shot_name.encode("utf-8"))
        hasher.update(str(version).encode("utf-8"))
        for f_idx, geom in enumerate(frames):
            hasher.update(str(f_idx).encode("utf-8"))
            hasher.update(geom.vertices.tobytes())
        return hasher.hexdigest()

    def verify_binding(self, current_cache_hash: str, current_version: int) -> bool:
        """Verify that this approval is cryptographically bound to the active result version and hash."""
        if self.approved_version != current_version:
            return False
        return self.cache_hash == current_cache_hash

    @classmethod
    def create_human_approval(
        cls,
        approved_version: int,
        cache_hash: str,
        artist_name: Optional[str] = None,
        notes: str = ""
    ) -> ApprovalRecord:
        """Create genuine human artist approval event via interactive operator signoff."""
        actor = artist_name or getpass.getuser() or "artist"
        return cls(
            authority=ApprovalAuthority.HUMAN_ARTIST,
            actor_id=actor,
            approved_version=approved_version,
            cache_hash=cache_hash,
            review_notes=notes or f"Approved interactively by artist '{actor}'",
            is_production_authorized=True,
        )

    @classmethod
    def create_synthetic_test_approval(
        cls,
        approved_version: int,
        cache_hash: str,
        test_case_id: str = "automated_test_runner",
        allow_isolated_publishing: bool = False
    ) -> ApprovalRecord:
        """Create synthetic test approval event, isolated from production."""
        return cls(
            authority=ApprovalAuthority.TEST_HARNESS_SYNTHETIC,
            actor_id=f"test_harness:{test_case_id}",
            approved_version=approved_version,
            cache_hash=cache_hash,
            review_notes=f"Synthetic test approval for verification suite '{test_case_id}'",
            is_production_authorized=allow_isolated_publishing,
        )


class ApprovalGatekeeper:
    """Enforces production publishing boundaries."""

    @staticmethod
    def validate_production_export_authorization(
        approval: Optional[ApprovalRecord],
        is_isolated_test_run: bool = False
    ) -> None:
        """Verify that an export is authorized for production publishing.

        Raises:
            PermissionError: If approval is missing or originated from a synthetic test
                outside an explicitly authorized isolated test run.
        """
        if approval is None:
            raise PermissionError("Export rejected: Simulation cache has not been approved.")

        if approval.authority == ApprovalAuthority.TEST_HARNESS_SYNTHETIC:
            if not is_isolated_test_run:
                raise PermissionError(
                    "Production delivery rejected: Cache was authorized only by a synthetic "
                    "test harness event. Genuine human artist review and approval is required "
                    "for production publishing."
                )

        if not approval.is_production_authorized and not is_isolated_test_run:
            raise PermissionError("Export rejected: Approval record is not authorized for production export.")
