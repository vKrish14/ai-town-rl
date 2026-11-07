"""Headless town state and transitions, independent of RL and rendering."""

from __future__ import annotations
from enum import IntEnum
import random
from datetime import date, timedelta
import numpy as np
from ai_town.citizen import Citizen
from ai_town.buildings import BUILDINGS, TOWN_WIDTH, TOWN_HEIGHT
from ai_town.economy import SharedEconomy
from ai_town.navigation import astar
from ai_town.rewards import DEFAULT_REWARD_WEIGHTS, calculate_reward
from ai_town.analytics import daily_town_snapshot


class Action(IntEnum):
    WORK = 0
    SLEEP = 1
    EAT = 2
    SOCIALISE = 3
    RECREATE = 4
    SHOP = 5
    EXERCISE = 6
    FIND_JOB = 7


ACTION_NAMES = [a.name.replace("_", " ").title() for a in Action]
OBS_DIM = 28
NAMES = [
    "Mira Chen", "Theo Alvarez", "Jun Park", "Asha Patel", "Felix Hart",
    "Nora Williams", "Ivo Mendes", "Lena Okafor", "Sam Rivera", "Zuri Bennett",
    "Arjun Mehta", "Priya Nair", "Elise Moreau", "Malik Johnson",
    "Talia Brooks", "Dev Shah",
    "Amara Okeke",
]
DEFAULT_POPULATION = len(NAMES)


class Town:
    width, height = TOWN_WIDTH, TOWN_HEIGHT
    buildings = BUILDINGS
    destinations = {
        Action.WORK: buildings["offices"],
        Action.SLEEP: buildings["homes"],
        Action.EAT: buildings["restaurant"] + buildings["homes"],
        Action.SOCIALISE: buildings["parks"] + buildings["town"],
        Action.RECREATE: buildings["parks"],
        Action.SHOP: buildings["groceries"],
        Action.EXERCISE: buildings["parks"],
        Action.FIND_JOB: buildings["offices"],
    }

    def __init__(
        self,
        population=DEFAULT_POPULATION,
        max_days=30,
        seed=None,
        reward_weights=None,
        ticks_per_day=24,
        economy=None,
        movement_steps_per_tick=3,
        town_events=None,
    ):
        if not 1 <= population <= len(NAMES):
            raise ValueError(f"population must be between 1 and {len(NAMES)}")
        self.population, self.max_days = population, max_days
        economy = dict(economy or {})
        economy.setdefault("shop_count", len(self.buildings["groceries"]))
        self.rng = random.Random(seed)
        self._event_rng = random.Random(None if seed is None else int(seed) ^ 0x5AFE71)
        self.town_events_config = dict(town_events or {})
        self.calendar_start = date.fromisoformat(self.town_events_config.get("calendar_start", "2026-01-01"))
        self.crime_config = dict(self.town_events_config.get("crime", {}))
        self._calendar_observances = {
            str(item.get("date", "")): str(item.get("name", "Community observance"))
            for item in self.town_events_config.get("observances", [])
        }
        self._observance_days_recorded: set[str] = set()
        self.tick = 0
        self.day = 0
        self.ticks_per_day = ticks_per_day
        if movement_steps_per_tick < 1:
            raise ValueError("movement_steps_per_tick must be positive")
        self.movement_steps_per_tick = movement_steps_per_tick
        self.hunger_drift = float(economy.get("hunger_drift_per_tick", 0.7))
        self.shop_daily_restock = int(economy.get("shop_daily_restock", 8))
        self.shared_economy = SharedEconomy.from_config(economy)
        self.clinic_visit_cost = float(economy.get("clinic_visit_cost", 10))
        self.events = []
        self._event_sequence = 0
        self.daily_summaries = []
        self.reward_weights = DEFAULT_REWARD_WEIGHTS | dict(reward_weights or {})
        self.citizens = [
            Citizen(
                i,
                NAMES[i],
                self.buildings["homes"][i],
                self.buildings["homes"][i],
                money=float(economy.get("starting_cash", 35)),
                food=int(economy.get("starting_food", 1)),
                personality=tuple(self.rng.random() for _ in range(4)),
                employed=i == len(NAMES) - 1,
                occupation="Police Officer" if i == len(NAMES) - 1 else "Unemployed",
                age=34 if i == len(NAMES) - 1 else 20,
            )
            for i in range(population)
        ]
        self.reward_components: dict[int, dict[str, float]] = {}
        # Day zero is the campaign's displayed Day 1; record its calendar and
        # ambient safety context as soon as the town is created.
        self._record_calendar_and_safety_events()

    @property
    def food_price(self):
        return self.shared_economy.food_price

    @property
    def food_stock(self):
        return self.shared_economy.food_stock

    @property
    def vacancies(self):
        return self.shared_economy.vacancies

    @vacancies.setter
    def vacancies(self, value):
        self.shared_economy.vacancies = value

    def _path(self, start, goal, occupied=()):
        obstacles = {
            p
            for group, cells in self.buildings.items()
            if group != "homes"
            for p in cells
        } | {p for p in self.buildings["homes"] if p != goal}
        return astar(start, goal, self.width, self.height, obstacles, set(occupied))

    def _access_points(self, kind):
        """Walkable entrance cells around each building footprint coordinate."""
        building_cells = {p for cells in self.buildings.values() for p in cells}
        points = []
        for x, y in self.buildings[kind]:
            for p in ((x, y + 1), (x - 1, y), (x + 1, y), (x, y - 1)):
                if (
                    0 <= p[0] < self.width
                    and 0 <= p[1] < self.height
                    and p not in building_cells
                ):
                    points.append(p)
        return points

    def _choose_destination(self, c, destinations):
        if c.position in destinations:
            return c.position

        reserved = set()
        for other in self.citizens:
            if other.id == c.id:
                continue
            reserved.add(other.position)
            if other.path and other.target is not None:
                reserved.add(other.target)
        unreserved = [point for point in destinations if point not in reserved]
        if unreserved:
            destinations = unreserved

        def score(point):
            distance = abs(point[0] - c.position[0]) + abs(point[1] - c.position[1])
            occupancy = sum(
                1
                for other in self.citizens
                if other.id != c.id
                and (other.target if other.path else other.position) == point
            )
            return distance + 4 * occupancy, distance, point

        return min(destinations, key=score)

    @staticmethod
    def _building_id(kind, index):
        prefixes = {
            "homes": "home", "offices": "office", "groceries": "grocery",
            "restaurant": "restaurant", "clinic": "clinic", "parks": "park",
            "town": "town_centre", "police": "police_station",
        }
        return f"{prefixes[kind]}_{index + 1:02d}"

    def _closest_building_id(self, kinds, position):
        candidates = [
            (abs(position[0] - x) + abs(position[1] - y), kind, index)
            for kind in kinds
            for index, (x, y) in enumerate(self.buildings[kind])
        ]
        if not candidates:
            return None
        _, kind, index = min(candidates)
        return self._building_id(kind, index)

    def _append_event(self, event, detail, citizen_id=None, building_id=None, reward=0, data=None):
        self._event_sequence += 1
        timestamp = {
            "day": self.tick // max(1, self.ticks_per_day) + 1,
            "hour": self.tick % max(1, self.ticks_per_day),
        }
        category = {
            "decision": "decision", "earning": "economy", "purchase": "economy",
            "clinic": "health", "employment": "employment", "survival": "health",
            "social": "social", "restock": "economy", "crime_report": "safety",
            "assault_report": "safety", "burglary_report": "safety",
            "robbery_report": "safety", "theft_report": "safety",
            "vandalism_report": "safety",
            "calendar_observance": "calendar",
        }.get(event, "town")
        row = {
            "event_id": f"event_{self._event_sequence:08d}",
            "tick": self.tick,
            "timestamp": timestamp,
            "category": category,
            "citizen": citizen_id,
            "citizen_id": citizen_id,
            "building_id": building_id,
            "event": event,
            "description": str(detail),
            "detail": detail,
            "data": dict(data or {}),
            "reward": float(reward),
        }
        self.events.append(row)
        return row

    def _record(self, c, event, detail, reward=0, building_id=None, data=None):
        row = self._append_event(event, detail, c.id, building_id, reward, data)
        c.record(row)

    def _record_finance(self, c, kind, amount, detail, building_id=None):
        row = {
            "tick": self.tick,
            "kind": kind,
            "amount": float(amount),
            "balance": float(c.money),
            "detail": detail,
        }
        c.financial_history.append(row)
        del c.financial_history[:-80]
        self._record(
            c, kind, detail, building_id=building_id,
            data={"amount": float(amount), "balance": float(c.money)},
        )

    @staticmethod
    def _wellbeing(c):
        return (
            c.health
            + c.energy
            + 100
            - c.hunger
            + c.happiness
            + 100
            - c.stress
            + c.social
        ) / 6

    def finalize_batch_rewards(self, baselines):
        """Score all agents against the same pre-step and post-step snapshots."""
        rewards = {}
        for c, (before, before_money, was_employed) in zip(self.citizens, baselines):
            if c.health <= 0:
                components = {name: 0.0 for name in self.reward_weights}
                components["survival"] = -float(self.reward_weights["survival"])
            else:
                components = calculate_reward(
                    before_wellbeing=before,
                    after_wellbeing=self._wellbeing(c),
                    hunger=c.hunger,
                    alive=True,
                    before_money=before_money,
                    money=c.money,
                    employed=c.employed,
                    was_employed=was_employed,
                    sustainable_work=c.employed
                    and c.money < 50
                    and c.activity == "Work",
                    weights=self.reward_weights,
                )
            reward = float(sum(components.values()))
            self.reward_components[c.id] = components
            rewards[f"citizen_{c.id}"] = reward
            if c.rewards:
                c.rewards[-1] = reward
            for row in reversed(c.history):
                if row["tick"] == self.tick and row["event"] == "decision":
                    row["reward"] = reward
                    break
        return rewards

    def step_agent(self, c, action):
        if c.health <= 0:
            c.activity = "Deceased"
            components = {name: 0.0 for name in self.reward_weights}
            components["survival"] = -float(self.reward_weights["survival"])
            self.reward_components[c.id] = components
            reward = float(components["survival"])
            c.rewards.append(reward)
            return reward
        before = self._wellbeing(c)
        before_money = c.money
        was_employed = c.employed
        try:
            requested = Action(int(action))
        except ValueError:
            requested = Action.RECREATE
        traveling = (
            c.pending_action is not None
            and c.target is not None
            and c.position != c.target
            and bool(c.path)
        )
        if traveling:
            act = Action(c.pending_action)
        else:
            act = requested
            c.pending_action = int(act)
        dests = [c.home] if act == Action.SLEEP else self.destinations[act]
        work_kind = "police" if c.occupation == "Police Officer" else "offices"
        if act == Action.WORK:
            dests = self._access_points(work_kind)
        elif act == Action.FIND_JOB:
            dests = self._access_points("offices")
        elif (
            act == Action.SLEEP and c.health < 35 and c.money >= self.clinic_visit_cost
        ):
            dests = self._access_points("clinic")
        elif act == Action.SHOP:
            dests = self._access_points("groceries")
        elif act == Action.EAT and c.food <= 0:
            dests = self._access_points("restaurant")
        elif act in (Action.SOCIALISE, Action.RECREATE, Action.EXERCISE):
            dests = self._access_points("parks") + self._access_points("town")
        if c.target is None or c.target not in dests or not c.path:
            c.target = self._choose_destination(c, dests)
            c.path = self._path(c.position, c.target)
        for _ in range(self.movement_steps_per_tick):
            if not c.path:
                break
            occupied = {x.position for x in self.citizens if x.id != c.id}
            if c.target in occupied:
                alternatives = [point for point in dests if point not in occupied]
                if alternatives:
                    c.target = self._choose_destination(c, alternatives)
                    c.path = self._path(c.position, c.target, occupied)
            if not c.path:
                break
            if c.path[0] in occupied:
                c.path = self._path(c.position, c.target, occupied)
            if not c.path or c.path[0] in occupied:
                break
            c.position = c.path.pop(0)
        arrived = c.position == c.target
        label = ACTION_NAMES[int(act)]
        if arrived:
            if (
                act == Action.WORK
                and c.employed
                and c.position in self._access_points(work_kind)
                and self.tick > 0
                and self.tick % max(1, self.ticks_per_day // 2) == 0
            ):
                if self.shared_economy.pay_salary(c, c.salary / 2):
                    self._record_finance(
                        c, "earning", c.salary / 2, f"Earned ${c.salary / 2:g} salary",
                        building_id=self._closest_building_id((work_kind,), c.position),
                    )
                c.energy -= 3
                c.hunger += 1.5
                c.stress += 0.5
            elif act == Action.SLEEP:
                c.energy += 24
                c.health += 1.5
                c.hunger += 1
                if (
                    c.position in self._access_points("clinic")
                    and c.money >= self.clinic_visit_cost
                ):
                    c.money -= self.clinic_visit_cost
                    c.health += 15
                    self._record_finance(
                        c,
                        "clinic",
                        -self.clinic_visit_cost,
                        f"Received treatment for ${self.clinic_visit_cost:g}",
                        building_id=self._closest_building_id(("clinic",), c.position),
                    )
            elif act == Action.EAT:
                if c.food > 0:
                    c.food -= 1
                    c.hunger -= 35
                elif c.position in self._access_points("restaurant") and c.money >= 6:
                    c.money -= 6
                    c.hunger -= 30
                    self._record_finance(
                        c, "purchase", -6, "Bought a restaurant meal",
                        building_id=self._closest_building_id(("restaurant",), c.position),
                    )
            elif act == Action.SOCIALISE:
                c.social += 18
                c.happiness += 7
                c.stress -= 4
                peers = [
                    other
                    for other in self.citizens
                    if other.id != c.id
                    and other.health > 0
                    and abs(other.position[0] - c.position[0])
                    + abs(other.position[1] - c.position[1])
                    <= 2
                ]
                if peers:
                    c.social += 4 + 4 * c.personality[1]
                    for peer in peers:
                        peer.social = min(100, peer.social + 3)
                        peer.happiness = min(100, peer.happiness + 2)
                    self._record(
                        c,
                        "social",
                        f"Spent time with {', '.join(peer.name for peer in peers)}",
                        building_id=self._closest_building_id(("parks", "town"), c.position),
                        data={"participants": [c.id, *(peer.id for peer in peers)]},
                    )
            elif act == Action.RECREATE:
                c.stress -= 12
                c.happiness += 7
                c.energy -= 3
            elif act == Action.SHOP:
                store = min(
                    range(len(self.food_stock)),
                    key=lambda i: abs(self.buildings["groceries"][i][0] - c.position[0])
                    + abs(self.buildings["groceries"][i][1] - c.position[1]),
                )
                if self.shared_economy.purchase_food(c, store):
                    self._record_finance(
                        c,
                        "purchase",
                        -self.food_price,
                        f"Bought food for ${self.food_price:g}",
                        building_id=self._building_id("groceries", store),
                    )
            elif act == Action.EXERCISE:
                c.health += 3
                c.energy -= 8
                c.hunger += 5
            elif (
                act == Action.FIND_JOB
                and c.position in self._access_points("offices")
                and self.shared_economy.hire(c)
            ):
                self._record(
                    c, "employment", "Started office work",
                    building_id=self._closest_building_id(("offices",), c.position),
                )
            c.activity = label
        else:
            c.activity = f"Walking to {label.lower()}"
        c.energy -= 1.4
        c.hunger += self.hunger_drift
        c.social -= 0.35
        if c.hunger > 80:
            c.health -= 0.8
        if c.energy < 15:
            c.health -= 0.45
        c.stress += 0.1
        for name in ("health", "energy", "hunger", "happiness", "stress", "social"):
            setattr(c, name, float(np.clip(getattr(c, name), 0, 100)))
        if c.health <= 0:
            self._record(c, "survival", "Citizen died")
        c.money = max(0, float(c.money))
        components = calculate_reward(
            before_wellbeing=before,
            after_wellbeing=self._wellbeing(c),
            hunger=c.hunger,
            alive=c.health > 0,
            before_money=before_money,
            money=c.money,
            employed=c.employed,
            was_employed=was_employed,
            sustainable_work=c.employed
            and c.money < 50
            and act == Action.WORK
            and arrived,
            weights=self.reward_weights,
        )
        reward = float(sum(components.values()))
        self.reward_components[c.id] = components
        c.rewards.append(reward)
        self._record(c, "decision", label, reward)
        return reward

    def advance_clock(self):
        self.tick += 1
        self.day = self.tick // self.ticks_per_day
        if self.tick % self.ticks_per_day == 0:
            before = list(self.food_stock)
            self.shared_economy.restock(self.shop_daily_restock)
            for index, (old_stock, new_stock) in enumerate(zip(before, self.food_stock)):
                if new_stock > old_stock:
                    self._append_event(
                        "restock",
                        f"Grocery stock replenished by {new_stock - old_stock}",
                        building_id=self._building_id("groceries", index),
                        data={"units": new_stock - old_stock, "stock": new_stock},
                    )
            self.daily_summaries.append(daily_town_snapshot(self, self.day))
            self._record_calendar_and_safety_events()

    @property
    def calendar_date(self):
        """Campaign date, advanced one calendar day per simulated day."""
        # day=0 is the live Day 1 shown immediately after reset.
        return self.calendar_start + timedelta(days=self.day)

    def _record_calendar_and_safety_events(self):
        """Add ambient civic telemetry without changing RL state or rewards."""
        today = self.calendar_date
        date_key = today.strftime("%m-%d")
        observance = self._calendar_observances.get(date_key)
        if observance and today.isoformat() not in self._observance_days_recorded:
            self._append_event(
                "calendar_observance", f"{observance} · {today.isoformat()}",
                building_id="town_centre_01",
                data={"calendar_date": today.isoformat(), "observance": observance},
            )
            self._observance_days_recorded.add(today.isoformat())
        if not self.crime_config.get("enabled", True):
            return
        rate = max(0.0, float(self.crime_config.get("rate_per_1000_resident_days", 12.0)))
        living = sum(citizen.alive for citizen in self.citizens)
        probability = min(1.0, rate * living / 1000.0)
        if living and self._event_rng.random() < probability:
            citizens = [citizen for citizen in self.citizens if citizen.alive]
            incident_weights = self.crime_config.get("incident_weights", {
                "assault_report": 0.30, "burglary_report": 0.28,
                "robbery_report": 0.18, "theft_report": 0.14,
                "vandalism_report": 0.10,
            })
            kinds = list(incident_weights)
            weights = [max(0.0, float(incident_weights[kind])) for kind in kinds]
            event_kind = self._event_rng.choices(kinds, weights=weights, k=1)[0]
            report_data = {
                "calendar_date": today.isoformat(),
                "status": "reported",
                "consequences_modeled": False,
                "narrative_only": True,
            }
            incident_building = "police_station_01"
            citizen_id = None
            if event_kind == "assault_report" and len(citizens) >= 2:
                suspect, victim = self._event_rng.sample(citizens, 2)
                detail = f"Assault reported: {suspect.name} attacked {victim.name}"
                citizen_id = victim.id
                report_data.update(suspect_id=suspect.id, victim_id=victim.id)
                incident_building = "town_centre_01"
            elif event_kind == "burglary_report":
                victim = self._event_rng.choice(citizens)
                home_index = min(victim.id, len(self.buildings["homes"]) - 1)
                detail = f"Home burglary reported at {victim.name}'s house"
                citizen_id = victim.id
                report_data.update(victim_id=victim.id, property_owner_id=victim.id)
                incident_building = self._building_id("homes", home_index)
            elif event_kind == "robbery_report":
                victim = self._event_rng.choice(citizens)
                detail = f"Robbery reported near Market on Juniper · {victim.name} filed a report"
                citizen_id = victim.id
                report_data["victim_id"] = victim.id
                incident_building = "grocery_03"
            elif event_kind == "vandalism_report":
                detail = "Vandalism reported on the Eastbank footbridge"
                incident_building = "park_02"
            else:
                detail = "Bicycle theft reported in the east quarter"
                incident_building = "police_station_01"
            self._append_event(
                event_kind, detail, citizen_id=citizen_id,
                building_id=incident_building, data=report_data,
            )

    def observation(self, c):
        other = (
            np.mean(
                [
                    abs(c.position[0] - x.position[0])
                    + abs(c.position[1] - x.position[1])
                    for x in self.citizens
                    if x.id != c.id
                ]
            )
            if len(self.citizens) > 1
            else 0
        )
        pending = np.zeros(8, dtype=np.float32)
        if c.pending_action is not None:
            pending[c.pending_action] = 1
        return np.asarray(
            [
                c.health / 100,
                c.energy / 100,
                c.hunger / 100,
                c.happiness / 100,
                c.stress / 100,
                c.social / 100,
                min(c.money / 200, 1),
                float(c.employed),
                (self.tick % self.ticks_per_day) / max(1, self.ticks_per_day - 1),
                # Preserve the normalized [0, 1] day feature after the viewer's
                # optional open-ended deployment passes its training horizon.
                min(1.0, self.day / max(1, self.max_days)),
                c.position[0] / 35,
                c.position[1] / 23,
                min(1, min(self.food_stock) / 40),
                min(1, self.vacancies / 4),
                min(1, c.food / 5),
                min(1, other / 50),
                *c.personality,
                *pending,
            ],
            dtype=np.float32,
        )

    def global_state(self):
        return np.concatenate([self.observation(c) for c in self.citizens]).astype(
            np.float32
        )
