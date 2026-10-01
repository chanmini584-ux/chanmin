"""Built-in synthetic scenarios: positive events + hard negatives (정상-유사 상황).

Cameras (see configs/site.demo.yaml):
  cam01 world x 0..20, cam02 x 26..46, cam03 x 52..72 (y 0..10 depth)
"""
from __future__ import annotations

from .scenario import Action, Actor, GTEvent, Hold, Keyframe as K, Scenario

RED, BLUE, BLACK, YELLOW = (40, 40, 200), (200, 90, 30), (35, 35, 35), (40, 200, 230)
GREEN, PURPLE, ORANGE, WHITE = (60, 170, 60), (150, 60, 140), (30, 140, 240), (230, 230, 230)
GRAY, PINK = (140, 140, 140), (180, 120, 230)


def assault_fight() -> Scenario:
    a = Actor("A", RED, [K(0, 28, 5), K(7, 35.4, 5), K(14.5, 35.4, 5), K(19, 50, 6), K(22, 58, 6)],
              actions=[Action(8, 14.5, "strike", "B")])
    b = Actor("B", BLUE, [K(0, 45, 5.2), K(7, 36.4, 5.2), K(30, 36.4, 5.2)],
              actions=[Action(9, 13, "strike", "A"), Action(14, 30, "fall")])
    w = Actor("W", GRAY, [K(0, 27, 1.5), K(30, 45, 1.5)])
    return Scenario(
        "assault_fight", "보행로 폭행 후 쓰러짐, 가해자 도주", 30, "2026-10-01T21:10:00+09:00",
        ["cam02", "cam03"], [a, b, w],
        [GTEvent("ASSAULT", 8, 14.5, ["cam02"], ["A", "B"]),
         GTEvent("FALL", 14, 30, ["cam02"], ["B"])],
    )


def knife_threat_chase() -> Scenario:
    a = Actor("A", BLACK, [K(0, 5, 4), K(6, 6, 4), K(9, 11.0, 4.6), K(10, 12.5, 4.6), K(20, 50, 5)],
              holds=[Hold(0, 35, "knife")])
    b = Actor("B", YELLOW, [K(0, 19, 5), K(6, 13.2, 4.8), K(9, 13.2, 4.8), K(9.6, 14, 4.8),
                            K(19, 52, 5.5), K(23, 66, 6)])
    c = Actor("C", GREEN, [K(0, 1, 8), K(35, 18, 8)])
    return Scenario(
        "knife_threat_chase", "흉기 소지자 접근 → 상대 도주 → 추격 (cam01→cam02→cam03)", 35,
        "2026-10-01T23:40:00+09:00", ["cam01", "cam02", "cam03"], [a, b, c],
        [GTEvent("WEAPON_THREAT", 6, 20, ["cam01", "cam02"], ["A", "B"]),
         GTEvent("CHASE", 9.6, 20, ["cam01", "cam02"], ["A", "B"])],
    )


def collapse_alone() -> Scenario:
    a = Actor("A", PURPLE, [K(0, 2, 6), K(7, 11.8, 6), K(30, 11.8, 6)],
              actions=[Action(7, 30, "fall")])
    p = Actor("P", GRAY, [K(0, 19, 2), K(30, 1, 2)])
    return Scenario("collapse_alone", "보행 중 단독 쓰러짐 (응급)", 30, "2026-10-01T14:05:00+09:00",
                    ["cam01"], [a, p], [GTEvent("FALL", 7, 30, ["cam01"], ["A"])])


def night_intrusion() -> Scenario:
    # restricted zone of cam03 covers image x >= 600 → world x >= 52 + 600/960*20 = 64.5
    a = Actor("A", BLACK, [K(0, 53, 3), K(8, 63, 5), K(10, 66, 6), K(18, 69, 4), K(26, 66.5, 7),
                           K(32, 63, 6), K(40, 54, 4)])
    return Scenario("night_intrusion", "심야 출입통제구역 침입", 40, "2026-10-02T02:15:00+09:00",
                    ["cam03"], [a], [GTEvent("INTRUSION", 9.0, 29.4, ["cam03"], ["A"])])


def loitering() -> Scenario:
    kf = [K(0, 26.5, 6)]
    t, x = 0.0, 26.5
    for i in range(9):  # back and forth inside the watch zone of cam02 (world x 30..40)
        x = 39 if i % 2 == 0 else 31
        t += 9.0
        kf.append(K(t, x, 6 + (i % 3) * 0.6))
        t += 3.0
        kf.append(K(t, x, 6 + (i % 3) * 0.6))
    kf.append(K(t + 6, 46.5, 6))
    a = Actor("A", ORANGE, kf)
    return Scenario("loitering", "심야 놀이터 인근 반복 배회", t + 6, "2026-10-01T23:55:00+09:00",
                    ["cam02"], [a], [GTEvent("LOITERING", 2.5, t + 0.8, ["cam02"], ["A"])])


# ---------------------------------------------------------------- hard negatives
def normal_walkers() -> Scenario:
    acts = [
        Actor("N1", RED, [K(0, 0, 3), K(60, 20, 3.5)]),
        Actor("N2", BLUE, [K(0, 20, 4), K(40, 0, 4.5)]),
        Actor("N3", GREEN, [K(5, 0, 7), K(25, 20, 6)]),
        Actor("N4", WHITE, [K(10, 20, 6.5), K(30, 0, 7)]),
        Actor("N5", PINK, [K(20, 0, 5), K(34, 20, 5), K(60, 20, 5)]),
        Actor("N6", YELLOW, [K(30, 20, 5.2), K(50, 0, 5.4)]),
    ]
    return Scenario("normal_walkers", "[정상] 일상 보행 (교차 통행)", 60, "2026-10-01T18:00:00+09:00",
                    ["cam01"], acts, hard_negative=True)


def jogging_pair() -> Scenario:
    a = Actor("A", RED, [K(0, 25, 5), K(10, 55, 5)])
    b = Actor("B", BLUE, [K(0, 25, 6.2), K(10, 55, 6.2)])
    c = Actor("C", GREEN, [K(12, 25, 4), K(22, 52, 4)])
    d = Actor("D", WHITE, [K(12.8, 25, 4.1), K(22.8, 52, 4.1)])  # follows C at ~2.2 m, same speed
    return Scenario("jogging_pair", "[정상] 나란히/앞뒤로 조깅", 25, "2026-10-01T07:00:00+09:00",
                    ["cam02"], [a, b, c, d], hard_negative=True,
                    note="추격(CHASE)과 유사한 '앞뒤 달리기' 포함")


def kitchen_knife_carry() -> Scenario:
    a = Actor("A", WHITE, [K(0, 0, 5), K(30, 20, 5)], holds=[Hold(0, 30, "knife")])
    b = Actor("B", BLUE, [K(3, 20, 8.5), K(25, 0, 8.5)])
    c = Actor("C", GREEN, [K(10, 20, 1.5), K(30, 2, 1.5)])
    return Scenario("kitchen_knife_carry", "[정상] 칼을 든 채 단독 보행 (접근·도주 없음)", 30,
                    "2026-10-01T12:00:00+09:00", ["cam01"], [a, b, c], hard_negative=True)


def greeting_hug() -> Scenario:
    a = Actor("A", RED, [K(0, 27, 5), K(7, 35.6, 5), K(13, 35.6, 5), K(25, 45, 5.5)])
    b = Actor("B", PINK, [K(0, 45, 5.2), K(7, 36.3, 5.2), K(13, 36.3, 5.2), K(25, 45.7, 5.7)])
    return Scenario("greeting_hug", "[정상] 만나서 가까이 서서 대화 후 함께 이동", 25,
                    "2026-10-01T19:00:00+09:00", ["cam02"], [a, b], hard_negative=True)


def phone_users() -> Scenario:
    acts = [
        Actor("P1", RED, [K(0, 0, 4), K(30, 20, 4)], holds=[Hold(0, 40, "phone")]),
        Actor("P2", BLUE, [K(0, 20, 4.8), K(30, 0, 4.8)], holds=[Hold(0, 40, "phone")]),
        Actor("P3", GREEN, [K(5, 0, 7), K(40, 20, 7)], holds=[Hold(0, 40, "phone")]),
        Actor("P4", YELLOW, [K(8, 20, 6.5), K(12, 14, 6.5), K(30, 14, 6.5), K(40, 20, 6.5)],
              holds=[Hold(0, 40, "phone")]),
    ]
    return Scenario("phone_users", "[정상] 휴대폰 사용 보행자 (흉기 오인 유발)", 40,
                    "2026-10-01T20:00:00+09:00", ["cam01"], acts, hard_negative=True)


def bat_sports() -> Scenario:
    a = Actor("A", GREEN, [K(0, 52, 5), K(25, 72, 5.5)], holds=[Hold(0, 25, "bat")])
    b = Actor("B", WHITE, [K(0, 52, 6.2), K(25, 72, 6.6)])
    return Scenario("bat_sports", "[정상] 야구 배트를 들고 친구와 함께 이동", 25,
                    "2026-10-01T16:00:00+09:00", ["cam03"], [a, b], hard_negative=True)


ALL = {f.__name__: f for f in [
    assault_fight, knife_threat_chase, collapse_alone, night_intrusion, loitering,
    normal_walkers, jogging_pair, kitchen_knife_carry, greeting_hug, phone_users, bat_sports,
]}


def get(name: str) -> Scenario:
    if name not in ALL:
        raise KeyError(f"unknown scenario '{name}'. available: {', '.join(ALL)}")
    return ALL[name]()
