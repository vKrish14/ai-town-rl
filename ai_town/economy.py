"""Shared finite supplies and atomic transactions."""

from dataclasses import dataclass
from ai_town.citizen import Citizen


@dataclass
class SharedEconomy:
    food_price: float = 5.0
    food_stock: list[int] | None = None
    vacancies: int = 4
    food_capacity: int = 40

    def __post_init__(self):
        if self.food_stock is None:
            self.food_stock = [40, 40]

    @classmethod
    def from_config(cls, config: dict | None = None):
        cfg = config or {}
        stock = int(cfg.get("shop_stock", 40))
        return cls(
            float(cfg.get("food_price", 5)),
            [stock] * max(1, int(cfg.get("shop_count", 2))),
            int(cfg.get("starting_vacancies", 4)),
            stock,
        )

    def purchase_food(self, citizen: Citizen, store: int) -> bool:
        if (
            self.food_stock is None
            or not 0 <= store < len(self.food_stock)
            or self.food_stock[store] <= 0
            or citizen.money < self.food_price
            or citizen.food >= 5
        ):
            return False
        citizen.money -= self.food_price
        citizen.food += 1
        self.food_stock[store] -= 1
        return True

    def restock(self, amount: int) -> None:
        """Replenish each shop up to its configured carrying capacity."""
        if amount < 0:
            raise ValueError("restock amount cannot be negative")
        if self.food_stock is not None:
            self.food_stock = [
                min(self.food_capacity, stock + amount) for stock in self.food_stock
            ]

    def hire(self, citizen: Citizen) -> bool:
        if citizen.employed or self.vacancies <= 0:
            return False
        citizen.employed = True
        citizen.occupation = "Office worker"
        self.vacancies -= 1
        return True

    @staticmethod
    def pay_salary(citizen: Citizen, amount: float) -> bool:
        if not citizen.employed or amount < 0:
            return False
        citizen.money += amount
        return True
