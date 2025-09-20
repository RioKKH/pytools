#!/usr/bin/env python


def main():
    V = {"L1": 0.0, "L2": 0.0}
    new_V = V.copy()  # Vのコピー

    cnt = 0  # 何回更新したかを記録
    while True:
        new_V["L1"] = 0.5 * (-1 + 0.9 * V["L1"]) + 0.5 * (1 + 0.9 * V["L2"])
        new_V["L2"] = 0.5 * (0 + 0.9 * V["L1"]) + 0.5 * (-1 + 0.9 * V["L2"])

        # 更新された量の最大値
        delta = abs(new_V["L1"] - V["L1"])
        delta = max(delta, abs(new_V["L2"] - V["L2"]))

        V = new_V.copy()  # Vを更新

        cnt += 1
        if delta < 0.0001:
            print(V)
            print(cnt)
            break


if __name__ == "__main__":
    main()
