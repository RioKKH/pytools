#!/usr/bin/env python


def main():
    V = {"L1": 0.0, "L2": 0.0}
    new_V = V.copy()  # Vのコピー

    for _ in range(100):
        new_V["L1"] = 0.5 * (-1 + 0.9 * V["L1"]) + 0.5 * (1 + 0.9 * V["L2"])
        new_V["L2"] = 0.5 * (0 + 0.9 * V["L1"]) + 0.5 * (-1 + 0.9 * V["L2"])

        V = new_V.copy()  # Vを更新
        print(V)


if __name__ == "__main__":
    main()
