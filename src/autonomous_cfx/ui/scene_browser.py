"""Autonomous CFX Studio - Scene Browser Widget.
Displays imported geometry hierarchy, selection, visibility toggles, and semantic roles.
Enforces explicit role confirmation (names alone do not dictate simulation roles).
"""

from __future__ import annotations

from typing import Dict, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from autonomous_cfx.core.alembic_io import AlembicNodeInfo, AlembicObjectRole


class SceneBrowserWidget(QWidget):
    """Hierarchy tree browser with explicit role assignment and validation warnings."""

    roles_confirmed = Signal(dict)          # Emits Dict[str, AlembicObjectRole]
    object_selected = Signal(str)           # Emits selected object path
    visibility_toggled = Signal(str, bool)  # Emits (object_path, is_visible)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.node_items: Dict[str, QTreeWidgetItem] = {}
        self.role_combos: Dict[str, QComboBox] = {}
        self.current_roles: Dict[str, AlembicObjectRole] = {}
        self.roles_are_confirmed: bool = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header Title
        lbl_title = QLabel("Scene Objects & Roles")
        lbl_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #f1f5f9;")
        layout.addWidget(lbl_title)

        # Tree Widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Object", "Role", "Verts", "Vis"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tree.setStyleSheet(
            "QTreeWidget { background-color: #11141a; border: 1px solid #232834; border-radius: 4px; color: #e2e8f0; }"
            "QTreeWidget::item:selected { background-color: #1e3a5f; color: #ffffff; }"
        )
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        layout.addWidget(self.tree, stretch=1)

        # Warnings / Summary Box
        self.warning_box = QLabel("No geometry loaded. Import an Alembic shot or load demo scene.")
        self.warning_box.setWordWrap(True)
        self.warning_box.setStyleSheet(
            "background-color: #1e222b; color: #94a3b8; font-size: 11px; padding: 6px; "
            "border-radius: 4px; border: 1px solid #2d3340;"
        )
        layout.addWidget(self.warning_box)

        # Confirm Button Row
        btn_row = QHBoxLayout()
        self.btn_confirm_roles = QPushButton("✔ Confirm Scene Roles")
        self.btn_confirm_roles.setEnabled(False)
        self.btn_confirm_roles.setStyleSheet(
            "background-color: #059669; color: white; font-weight: bold; padding: 8px; border-radius: 4px;"
        )
        self.btn_confirm_roles.clicked.connect(self._on_confirm_clicked)
        btn_row.addWidget(self.btn_confirm_roles)

        layout.addLayout(btn_row)

    def populate_nodes(self, nodes: List[AlembicNodeInfo]):
        """Populate tree with inspected Alembic nodes."""
        self.tree.clear()
        self.node_items.clear()
        self.role_combos.clear()
        self.current_roles.clear()
        self.roles_are_confirmed = False
        self.btn_confirm_roles.setEnabled(len(nodes) > 0)

        poly_nodes = [n for n in nodes if n.node_type == "PolyMesh"]

        for node in poly_nodes:
            item = QTreeWidgetItem(self.tree)
            item.setText(0, f"📦 {node.name}")
            item.setToolTip(0, f"Full Path: {node.full_path}\nSamples: {node.num_samples}")
            item.setText(2, str(node.num_vertices))
            item.setData(0, Qt.UserRole, node.full_path)

            # Role combo
            combo = QComboBox()
            combo.addItems(["Garment", "Body Collider", "Prop Collider", "Ignored"])
            combo.setStyleSheet("font-size: 11px; padding: 2px 4px;")

            # Initial inference suggestion (not confirmed until clicked!)
            suggested_role = node.assigned_role.value if node.assigned_role else "ignored"
            if "garment" in suggested_role or "cloth" in node.name.lower() or "jacket" in node.name.lower():
                combo.setCurrentText("Garment")
                self.current_roles[node.full_path] = AlembicObjectRole.GARMENT
            elif "body" in suggested_role or "char" in node.name.lower() or "body" in node.name.lower():
                combo.setCurrentText("Body Collider")
                self.current_roles[node.full_path] = AlembicObjectRole.BODY
            elif "chair" in node.name.lower() or "table" in node.name.lower() or "prop" in suggested_role:
                combo.setCurrentText("Prop Collider")
                self.current_roles[node.full_path] = AlembicObjectRole.PROP
            else:
                combo.setCurrentText("Ignored")
                self.current_roles[node.full_path] = AlembicObjectRole.IGNORED

            path_key = node.full_path
            combo.currentTextChanged.connect(lambda text, p=path_key: self._on_role_changed(p, text))
            self.role_combos[path_key] = combo

            self.tree.setItemWidget(item, 1, combo)

            # Visibility Checkbox
            item.setCheckState(3, Qt.Checked)

            # Topology check
            if not node.is_topology_constant:
                item.setText(0, f"⚠️ {node.name} (Changing Topology)")
                item.setToolTip(0, "Warning: Changing vertex count across animation is unsupported.")

            self.node_items[path_key] = item

        self.tree.itemChanged.connect(self._on_item_changed)
        self._update_warnings()

    def set_objects_from_geometries(self, garment_name: str, garment_verts: int, colliders: List[tuple[str, str, int]]):
        """Populate tree directly from in-memory objects (e.g. Demo Scene)."""
        nodes = [
            AlembicNodeInfo(
                name=garment_name,
                full_path=f"/shot/{garment_name}",
                node_type="PolyMesh",
                parent_path="/shot",
                assigned_role=AlembicObjectRole.GARMENT,
                num_vertices=garment_verts,
                num_samples=40,
            )
        ]
        for c_name, c_role, c_verts in colliders:
            r = AlembicObjectRole.BODY if c_role == "body" else AlembicObjectRole.PROP
            nodes.append(
                AlembicNodeInfo(
                    name=c_name,
                    full_path=f"/shot/{c_name}",
                    node_type="PolyMesh",
                    parent_path="/shot",
                    assigned_role=r,
                    num_vertices=c_verts,
                    num_samples=40,
                )
            )
        self.populate_nodes(nodes)

    def _on_role_changed(self, full_path: str, role_text: str):
        self.roles_are_confirmed = False
        role_map = {
            "Garment": AlembicObjectRole.GARMENT,
            "Body Collider": AlembicObjectRole.BODY,
            "Prop Collider": AlembicObjectRole.PROP,
            "Ignored": AlembicObjectRole.IGNORED,
        }
        self.current_roles[full_path] = role_map.get(role_text, AlembicObjectRole.IGNORED)
        self._update_warnings()

    def _on_confirm_clicked(self):
        self.roles_are_confirmed = True
        self._update_warnings()
        self.roles_confirmed.emit(self.current_roles)

    def _on_tree_selection_changed(self):
        items = self.tree.selectedItems()
        if items:
            path = items[0].data(0, Qt.UserRole)
            if path:
                self.object_selected.emit(path)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        if column == 3:
            path = item.data(0, Qt.UserRole)
            is_vis = (item.checkState(3) == Qt.Checked)
            if path:
                self.visibility_toggled.emit(path, is_vis)

    def _update_warnings(self):
        garment_count = sum(1 for r in self.current_roles.values() if r == AlembicObjectRole.GARMENT)
        body_count = sum(1 for r in self.current_roles.values() if r == AlembicObjectRole.BODY)
        prop_count = sum(1 for r in self.current_roles.values() if r == AlembicObjectRole.PROP)

        if not self.current_roles:
            self.warning_box.setText("No geometry loaded.")
            self.warning_box.setStyleSheet(
                "background-color: #1e222b; color: #94a3b8; font-size: 11px; padding: 6px; border-radius: 4px;"
            )
            return

        warnings = []
        if garment_count == 0:
            warnings.append("⚠️ No Garment role assigned. At least 1 garment is required.")
        elif garment_count > 1:
            warnings.append(f"ℹ️ {garment_count} Garments selected (Multi-garment simulation).")

        if body_count == 0:
            warnings.append("⚠️ No Body Collider assigned. Cloth will have no character collision.")

        if not self.roles_are_confirmed:
            warnings.append("⏳ Roles modified. Click 'Confirm Scene Roles' to commit.")
            self.warning_box.setStyleSheet(
                "background-color: #3b2808; color: #fbbf24; font-size: 11px; padding: 6px; "
                "border-radius: 4px; border: 1px solid #78350f;"
            )
        else:
            self.warning_box.setStyleSheet(
                "background-color: #064e3b; color: #34d399; font-size: 11px; padding: 6px; "
                "border-radius: 4px; border: 1px solid #047857;"
            )

        summary_msg = f"{garment_count} Garment(s), {body_count} Body Collider(s), {prop_count} Prop(s)."
        if warnings:
            self.warning_box.setText(f"{summary_msg}\n" + "\n".join(warnings))
        else:
            self.warning_box.setText(f"✓ {summary_msg} Roles confirmed. Ready to configure behavior.")

    def has_valid_roles(self) -> bool:
        """Check whether at least one garment is confirmed."""
        if not self.roles_are_confirmed:
            return False
        return any(r == AlembicObjectRole.GARMENT for r in self.current_roles.values())
