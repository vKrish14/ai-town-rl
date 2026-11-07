from ai_town.config import load_config
from ai_town.simulation import Town
from ai_town.town_info import (
    building_at_world_position,
    building_catalog,
    building_visual_bounds,
    inspect_building,
    town_identity,
    town_statistics,
)


def test_named_building_catalog_has_unique_stable_ids_and_configurable_names():
    config = load_config()
    buildings = building_catalog(config)
    assert len(buildings) == 33
    assert len({item.id for item in buildings}) == len(buildings)
    assert len({item.name for item in buildings}) == len(buildings)
    assert next(item for item in buildings if item.id == "grocery_01").name == "GreenBasket Market"
    # Display and simulation lots now share the same road-cleared footprint.
    home_three = next(item for item in buildings if item.id == "home_03")
    assert home_three.position == home_three.simulation_position == (19, 3)


def test_building_hit_test_uses_the_same_bounds_as_selection_outline():
    catalog = building_catalog(load_config())
    park_sizes = set()
    for building in catalog:
        left, top, right, bottom = building_visual_bounds(building)
        size = (right - left, bottom - top)
        if building.kind == "parks":
            park_sizes.add(size)
        elif building.kind == "offices":
            assert size == (88 / 32, 76 / 32)
        elif building.kind == "homes" and building.index % 3 == 1:
            assert size == (84 / 32, 82 / 32)
        elif building.kind == "homes":
            assert size == (84 / 32, 76 / 32)
        else:
            assert size == (84 / 32, 76 / 32)
        assert building_at_world_position(catalog, (left + right) / 2, (top + bottom) / 2) == building
    assert park_sizes == {(6, 5)}


def test_building_inspectors_use_real_state_and_mark_untracked_fields():
    town = Town(population=2, seed=3)
    catalog = building_catalog(load_config())
    grocery = inspect_building(town, next(item for item in catalog if item.id == "grocery_01"))
    assert grocery["food_stock"] == town.food_stock[0]
    assert grocery["food_price"] == town.food_price
    assert grocery["capacity"] is None

    restaurant = inspect_building(town, next(item for item in catalog if item.id == "restaurant_01"))
    assert restaurant["food_inventory"] is None
    assert restaurant["inventory_status"] == "Not tracked by the simulation"

    home = inspect_building(town, next(item for item in catalog if item.id == "home_01"))
    assert home["residents"] == [town.citizens[0].name]
    assert home["household"][0]["wealth"] == town.citizens[0].money


def test_police_station_lists_the_rl_controlled_officer():
    town = Town(seed=3)
    catalog = building_catalog(load_config())
    station = inspect_building(town, next(item for item in catalog if item.id == "police_station_01"))
    assert "Amara Okeke" in station["officers"]
    assert "dispatch is not modeled" in station["status"]


def test_structured_events_have_unique_ids_real_timestamps_and_building_links():
    town = Town(population=1, seed=4)
    citizen = town.citizens[0]
    town._record(citizen, "employment", "Started office work", building_id="office_01")
    event = town.events[-1]
    assert event["event_id"] == "event_00000001"
    assert event["timestamp"] == {"day": 1, "hour": 0}
    assert event["category"] == "employment"
    assert event["citizen_id"] == citizen.id
    assert event["building_id"] == "office_01"
    assert event["description"] == "Started office work"

    town.tick = 23
    town.food_stock[:] = [10, 40]
    town.advance_clock()
    replenishment = town.events[-1]
    assert replenishment["event"] == "restock"
    assert replenishment["building_id"] == "grocery_01"
    assert replenishment["data"]["units"] == 8
    assert replenishment["timestamp"] == {"day": 2, "hour": 0}
    assert event["event_id"] != replenishment["event_id"]


def test_town_statistics_are_derived_from_citizens_and_event_data():
    town = Town(population=2, seed=5)
    town.citizens[0].money = 20
    town.citizens[1].money = 40
    town.citizens[0].employed = True
    town._append_event("earning", "Salary", 0, "office_01", data={"amount": 6})
    town._append_event("purchase", "Food", 1, "grocery_02", data={"amount": -5})
    town._append_event("social", "Talked", 0, "park_01", data={"participants": [0, 1]})
    stats = town_statistics(town)
    assert stats["population"] == 2
    assert stats["employment_rate"] == 0.5
    assert stats["wealth_total"] == 60
    assert stats["salary_payments_total"] == 6
    assert stats["shopping_spend_total"] == 5
    assert stats["social_interactions_recorded"] == 1


def test_map_hit_testing_and_town_identity_use_configured_data():
    config = load_config()
    town = Town(population=1, seed=8)
    catalog = building_catalog(config)
    assert building_at_world_position(catalog, 4.5, 13.5).id == "grocery_01"
    assert building_at_world_position(catalog, 15.5, 9.5) is None
    identity = town_identity(config, town)
    assert identity["name"] == "Willowbrook"
    assert identity["simulated_age_days"] == 1
    assert identity["weather"] == "Not simulated"


def test_day_boundary_keeps_a_snapshot_of_real_state_and_recorded_activity():
    town = Town(population=2, seed=9)
    town.tick = 23
    town.citizens[0].money = 25
    town.citizens[0].employed = True
    town.food_stock[:] = [10, 40]
    town._append_event("earning", "Salary", 0, "office_01", data={"amount": 6})
    town._append_event("purchase", "Food", 1, "grocery_01", data={"amount": -5})
    town._append_event("social", "Conversation", 0, "park_01")
    town.advance_clock()
    assert len(town.daily_summaries) == 1
    summary = town.daily_summaries[0]
    assert summary["day"] == 1
    assert summary["population"] == 2
    assert summary["employment_rate"] == 0.5
    assert summary["salary_payments"] == 6
    assert summary["shopping_spend"] == 5
    assert summary["social_interactions"] == 1
    assert summary["food_stock"] == sum(town.food_stock) == 58
