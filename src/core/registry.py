from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Component(Protocol):
    """
    Standardized Component Protocol (Section 5).
    Every switchable agent or environment module implements `name`, `version`,
    `config`, and `describe()`, and is selected exclusively via config keys.
    """

    name: str
    version: str
    config: Dict[str, Any]

    def describe(self) -> Dict[str, Any]:
        ...


class BaseRegisteredComponent:
    """Base helper implementing the `Component` protocol."""

    def __init__(
        self,
        name: str,
        variant: str,
        version: str = "1.0.0",
        config: Dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.variant = variant
        self.version = version
        self.config = config or {}

    def describe(self) -> Dict[str, Any]:
        return {
            "component": self.name,
            "variant": self.variant,
            "version": self.version,
            "is_null": self.variant in ("none", "off"),
            "config": self.config,
        }


class ComponentRegistry:
    """
    Central registry for all 11 switchable agent component families and baselines.
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Dict[str, Callable[..., Component]]] = {}

    def register(
        self,
        component_family: str,
        variant_name: str,
        factory: Callable[..., Component],
    ) -> None:
        fam = component_family.strip().lower()
        var = variant_name.strip().lower()
        self._factories.setdefault(fam, {})[var] = factory

    def build(
        self,
        component_family: str,
        variant_name: str,
        **kwargs: Any,
    ) -> Component:
        fam = component_family.strip().lower()
        var = variant_name.strip().lower()
        if fam not in self._factories or var not in self._factories[fam]:
            return BaseRegisteredComponent(
                name=fam, variant=var, config=kwargs
            )
        return self._factories[fam][var](**kwargs)

    def list_variants(self, component_family: str) -> List[str]:
        fam = component_family.strip().lower()
        return sorted(self._factories.get(fam, {}).keys())


GLOBAL_COMPONENT_REGISTRY = ComponentRegistry()