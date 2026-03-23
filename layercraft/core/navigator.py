"""DataNavigator - understands and traverses the alternating layer structure.

Layer alternation pattern
--------------------------
- **Odd levels  (1, 3, 5, …)**: Attribute/property layers – keys are fixed
  property names such as ``"samples"``, ``"day"``, ``"abundance"``.
- **Even levels (2, 4, 6, …)**: ID/classification layers – keys are variable
  instance identifiers such as ``"A1_1"``, ``"OTU1"``.
- **Leaf nodes**: Always at odd-numbered layers (scalar values).
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Tuple


PathKey = Tuple[str, ...]


class DataNavigator:
    """Navigate and query an alternating-layer hierarchical data structure.

    Parameters
    ----------
    data:
        The root dict of the data structure (level 1 attribute layer).
    entities_meta:
        Optional pre-computed entities metadata (output of
        ``analyze_structure.py``).  When provided, entity lookup is O(1).
    """

    def __init__(
        self,
        data: Dict[str, Any],
        entities_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        self._data = data
        self._meta: Dict[str, Any] = entities_meta or {}
        # entity_path_display → entity record
        self._entity_index: Dict[str, Dict[str, Any]] = {
            e["entity_path_display"]: e
            for e in self._meta.get("entities", [])
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def data(self) -> Dict[str, Any]:
        """Return the underlying data dict."""
        return self._data

    def get_entity_meta(self, entity_path_display: str) -> Optional[Dict[str, Any]]:
        """Return the metadata record for an entity given its display path.

        Parameters
        ----------
        entity_path_display:
            e.g. ``"samples"`` or ``"samples > bacteria"``.
        """
        return self._entity_index.get(entity_path_display)

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def iter_entity_instances(
        self,
        entity_path: PathKey,
    ) -> Generator[Tuple[List[str], Dict[str, Any]], None, None]:
        """Yield ``(id_chain, attribute_dict)`` for every instance of an entity.

        Parameters
        ----------
        entity_path:
            Tuple of collection-key names leading to the entity, e.g.
            ``("samples",)`` or ``("samples", "bacteria")``.

        Yields
        ------
        id_chain:
            List of IDs from the root down to this instance,
            e.g. ``["A1_1"]`` or ``["A1_1", "OTU1"]``.
        attribute_dict:
            The attribute dict for this instance (odd-level node).
        """
        yield from self._iter_instances(
            node=self._data,
            remaining_path=list(entity_path),
            id_chain=[],
        )

    def _iter_instances(
        self,
        node: Dict[str, Any],
        remaining_path: List[str],
        id_chain: List[str],
    ) -> Generator[Tuple[List[str], Dict[str, Any]], None, None]:
        if not remaining_path:
            yield id_chain, node
            return

        collection_key = remaining_path[0]
        rest = remaining_path[1:]

        collection = node.get(collection_key)
        if not isinstance(collection, dict):
            return

        for instance_id, payload in collection.items():
            if isinstance(payload, dict):
                yield from self._iter_instances(
                    node=payload,
                    remaining_path=rest,
                    id_chain=id_chain + [instance_id],
                )

    # ------------------------------------------------------------------
    # Property access
    # ------------------------------------------------------------------

    def get_property(
        self,
        entity_path: PathKey,
        instance_id_chain: List[str],
        property_name: str,
    ) -> Any:
        """Return the value of *property_name* for a specific instance.

        Searches the instance's own attribute dict first, then walks up the
        ancestor chain looking for inherited properties.

        Parameters
        ----------
        entity_path:
            e.g. ``("samples", "bacteria")``.
        instance_id_chain:
            e.g. ``["A1_1", "OTU1"]``.
        property_name:
            The attribute key to retrieve.

        Returns
        -------
        The value, or ``None`` if not found anywhere in the hierarchy.
        """
        # Traverse down to the instance node
        node = self._data
        for collection_key, instance_id in zip(entity_path, instance_id_chain):
            collection = node.get(collection_key)
            if not isinstance(collection, dict):
                return None
            node = collection.get(instance_id)
            if not isinstance(node, dict):
                return None

        # Walk back up the path collecting ancestor attribute dicts
        # Build a stack: deepest first, then shallower
        ancestor_nodes: List[Dict[str, Any]] = [node]
        walker = self._data
        for depth, (collection_key, instance_id) in enumerate(
            zip(entity_path, instance_id_chain)
        ):
            collection = walker.get(collection_key)
            if not isinstance(collection, dict):
                break
            ancestor_instance = collection.get(instance_id)
            if depth < len(entity_path) - 1 and isinstance(ancestor_instance, dict):
                ancestor_nodes.append(ancestor_instance)
            walker = ancestor_instance if isinstance(ancestor_instance, dict) else {}

        # Search deepest → shallowest
        for attr_node in reversed(ancestor_nodes):
            if property_name in attr_node:
                return attr_node[property_name]

        # Also check root level
        if property_name in self._data:
            return self._data[property_name]

        return None

    def set_property(
        self,
        entity_path: PathKey,
        instance_id_chain: List[str],
        property_name: str,
        value: Any,
    ) -> None:
        """Write *value* into the attribute dict of the given instance.

        Parameters
        ----------
        entity_path, instance_id_chain:
            Identify the target instance (same semantics as
            :meth:`get_property`).
        property_name:
            The attribute key to set.
        value:
            The value to store.
        """
        node = self._data
        for collection_key, instance_id in zip(entity_path, instance_id_chain):
            collection = node.get(collection_key)
            if not isinstance(collection, dict):
                raise KeyError(
                    f"Collection key '{collection_key}' not found in node"
                )
            node = collection.get(instance_id)
            if not isinstance(node, dict):
                raise KeyError(
                    f"Instance ID '{instance_id}' not found under '{collection_key}'"
                )
        node[property_name] = value

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def collect_property_values(
        self,
        entity_path: PathKey,
        property_name: str,
        inherited: bool = False,
    ) -> Dict[str, Any]:
        """Collect *property_name* for every instance of an entity.

        Parameters
        ----------
        entity_path:
            e.g. ``("samples", "bacteria")``.
        property_name:
            Attribute key to collect.
        inherited:
            If *True*, also search ancestor attribute dicts when the
            property is not found on the instance itself.

        Returns
        -------
        dict mapping ``str(id_chain)`` → value for instances where the
        property is present.
        """
        result: Dict[str, Any] = {}
        for id_chain, attr_dict in self.iter_entity_instances(entity_path):
            if property_name in attr_dict:
                result[str(id_chain)] = attr_dict[property_name]
            elif inherited:
                val = self.get_property(entity_path, id_chain, property_name)
                if val is not None:
                    result[str(id_chain)] = val
        return result

    def collect_property_series(
        self,
        entity_path: PathKey,
        property_name: str,
        inherited: bool = False,
    ) -> Tuple[List[List[str]], List[Any]]:
        """Like :meth:`collect_property_values` but returns parallel lists.

        Returns
        -------
        id_chains:
            List of id-chain lists (one per instance).
        values:
            Corresponding property values.
        """
        id_chains: List[List[str]] = []
        values: List[Any] = []
        for id_chain, attr_dict in self.iter_entity_instances(entity_path):
            if property_name in attr_dict:
                id_chains.append(id_chain)
                values.append(attr_dict[property_name])
            elif inherited:
                val = self.get_property(entity_path, id_chain, property_name)
                if val is not None:
                    id_chains.append(id_chain)
                    values.append(val)
        return id_chains, values

    # ------------------------------------------------------------------
    # Structural inspection
    # ------------------------------------------------------------------

    def list_entity_paths(self) -> List[str]:
        """Return display paths of all known entities (from metadata)."""
        return list(self._entity_index.keys())

    def resolve_entity_path(self, display: str) -> Optional[PathKey]:
        """Convert a display path ``"samples > bacteria"`` to a tuple.

        Falls back to splitting on ``" > "`` when the metadata index does
        not contain the display string.
        """
        meta = self._entity_index.get(display)
        if meta is not None:
            return tuple(meta["entity_path"])
        # Fallback: split on " > "
        parts = [p.strip() for p in display.split(" > ")]
        return tuple(parts) if parts else None
