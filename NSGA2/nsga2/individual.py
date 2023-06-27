#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Individual:

    def __init__(self):
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None
        self.objectives = None

    def __eq__(self, other):
        """
        __eq__はPythonの特殊メソッドの一つで、オブジェクトの等価性を比較する
        のに使われる。このメソッドを定義することで、オブジェクト同士を==演算子
        で比較できるようになる
        """
        if isinstance(self, other.__class__):
            # selfとotherが同じクラスのインスタンスかどうかを確認するために使用
            # さらにfeaturesの値が等しい場合に等価とみなす、としている
            return self.features == other.features
        return False

    def dominates(self, other_individual):
        """
        Check if self dominates other.
        self dominates other is true if:
        1. self is no worse than other in all objectives.
        2. self is strictly better than other in at least one objectives.
        """
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition  = or_condition  or  first <  second
        return (and_condition and or_condition)

        # このようにも書けるはず。上の書き方が正常に動作したのち、試してみる事
            #not_worse_in_all = all(first <= second)
            #better_in_one    = any(first <  second)
        #return (not_worse_in_all and better_in_one)
