import numpy as np
from ai_town.simulation import Town, Action


def test_bounds_transactions_and_navigation():
    town = Town(10, 3, 3)
    c = town.citizens[0]
    c.money = 20
    for _ in range(60):
        town.step_agent(c, Action.SOCIALISE)
        town.advance_clock()
    assert all(
        0 <= getattr(c, k) <= 100
        for k in ("health", "energy", "hunger", "happiness", "stress", "social")
    )
    assert c.money >= 0
    assert town._path(c.position, c.home) or c.position == c.home


def test_food_purchase_is_atomic():
    town = Town(1, 2, 1)
    c = town.citizens[0]
    c.money = 0
    before = town.food_stock[:]
    for _ in range(15):
        town.step_agent(c, Action.SHOP)
    assert c.money == 0 and c.food == 1 and town.food_stock == before


def test_shops_restock_daily_up_to_capacity():
    town = Town(1, 2, 1, ticks_per_day=2, economy={"shop_stock": 5})
    town.shared_economy.food_stock[0] = 1
    town.advance_clock()
    assert town.food_stock[0] == 1
    town.advance_clock()
    assert town.food_stock == [5, 5, 5]


def test_salary_records_amount_and_resulting_balance():
    town = Town(1, 2, 1)
    citizen = town.citizens[0]
    citizen.employed = True
    citizen.money = 10
    citizen.position = town._access_points("offices")[0]
    town.tick = 12
    town.step_agent(citizen, Action.WORK)
    assert citizen.money == 16
    assert citizen.financial_history[-1]["kind"] == "earning"
    assert citizen.financial_history[-1]["amount"] == 6
    assert citizen.financial_history[-1]["balance"] == 16


def test_grocery_purchase_records_the_debit_and_balance():
    town = Town(1, 2, 1)
    citizen = town.citizens[0]
    citizen.position = town._access_points("groceries")[0]
    citizen.money = 20
    town.step_agent(citizen, Action.SHOP)
    transaction = citizen.financial_history[-1]
    assert transaction["kind"] == "purchase"
    assert transaction["amount"] == -town.food_price
    assert transaction["balance"] == citizen.money == 15


def test_cannot_eat_a_paid_meal_from_home():
    town = Town(1, 2, 1)
    c = town.citizens[0]
    c.food = 0
    c.money = 20
    for _ in range(3):
        town.step_agent(c, Action.EAT)
    assert c.money == 20 and c.hunger > 20


def test_action_bounds_and_finite_rewards():
    town = Town(1, 2, 1)
    c = town.citizens[0]
    assert np.isfinite(town.step_agent(c, 999))


def test_deceased_citizen_does_not_keep_acting():
    town = Town(1, 2, 1)
    c = town.citizens[0]
    c.health = 0
    before = (c.money, c.position)
    assert town.step_agent(c, Action.WORK) == -town.reward_weights["survival"]
    assert (c.money, c.position) == before and c.activity == "Deceased"


def test_path_movement_progresses_and_job_action_completes():
    town = Town(1, 3, 4)
    citizen = town.citizens[0]
    for _ in range(40):
        town.step_agent(citizen, Action.FIND_JOB)
        if citizen.employed:
            break
        town.advance_clock()
    assert citizen.employed
    assert citizen.position in town._access_points("offices")


def test_route_retargets_when_another_citizen_occupies_entrance():
    town = Town(2, 3, 11)
    citizen, blocker = town.citizens
    citizen.position = (25, 9)
    blocker.position = (26, 9)
    town.step_agent(citizen, Action.WORK)
    assert citizen.target != blocker.position
    assert citizen.position != blocker.position
    assert citizen.position != (25, 9)


def test_police_officer_is_a_policy_citizen_with_station_workplace():
    town = Town(seed=19)
    officer = town.citizens[-1]
    station_entrances = town._access_points("police")
    assert officer.name == "Amara Okeke"
    assert officer.occupation == "Police Officer" and officer.employed
    assert officer.home == town.buildings["homes"][-1]
    town.step_agent(officer, Action.WORK)
    assert officer.target in station_entrances


def test_clinic_treatment_is_a_paid_building_interaction():
    town = Town(1, 3, 4)
    citizen = town.citizens[0]
    citizen.health = 20
    citizen.money = 50
    for _ in range(40):
        town.step_agent(citizen, Action.SLEEP)
        if any(event["event"] == "clinic" for event in citizen.history):
            break
        town.advance_clock()
    assert citizen.money == 40
    assert any(event["event"] == "clinic" for event in citizen.history)
    assert citizen.financial_history[-1]["kind"] == "clinic"
    assert citizen.financial_history[-1]["amount"] == -10


def test_socialising_near_a_neighbor_improves_both_citizens():
    town = Town(2, 2, 4)
    citizen, neighbor = town.citizens
    meeting_point = town._access_points("town")[0]
    citizen.position = meeting_point
    neighbor.position = meeting_point
    old_social = neighbor.social
    town.step_agent(citizen, Action.SOCIALISE)
    assert neighbor.social > old_social
