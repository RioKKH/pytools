#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np


class UnicarsalLine(object):
    """ This is a class to make a unicursal line which spreads out like circle
    from center of itself and go out of itself step by step.

    : param last_layer: For constructor. Need to specify the last layer to stop
                        generating flow.
    """

    def __init__(self, last_layer):
        """ Constructor of UnicarsalLine.

        :param current: current position
        :param next: candidate position to move next
        """
        self.status = 0
        self.last_layer = last_layer
        self.current_layer = 0
        self.current_pos = [0, 0]

    def show_current_position(self):
        print("%f %f"
              % (self.current_pos[0], self.current_pos[1]))

    def get_current_position(self):
        return self.current_pos[0], self.current_pos[1]

    def get_next(self):
        """ Get a next position and return it to a caller.

        :param status 0: Moving up
        :param status 1: Moving right
        :param status 2: Moving down
        :param status 3: Moving left
        """
        if self.status == 0:
            self.current_pos[1] += 1
            if np.abs(self.current_pos[1]) > self.current_layer:
                self.current_layer += 1
                self.status = 1
                if self.current_layer == self.last_layer:
                    return 1
        elif self.status == 1:
            self.current_pos[0] += 1
            if np.abs(self.current_pos[0]) == self.current_layer:
                self.status = 2
        elif self.status == 2:
            self.current_pos[1] -= 1
            if np.abs(self.current_pos[1]) == self.current_layer:
                self.status = 3
        elif self.status == 3:
            self.current_pos[0] -= 1
            if np.abs(self.current_pos[0]) == self.current_layer:
                self.status = 0

    def get_list(self):
        posx = []
        posy = []
        while True:
            if self.status == 0:
                self.current_pos[1] += 1
                posx.append(self.current_pos[0])
                posy.append(self.current_pos[1])
                if np.abs(self.current_pos[1]) > self.current_layer:
                    self.current_layer += 1
                    self.status = 1
                    if self.current_layer == self.last_layer:
                        break
            elif self.status == 1:
                self.current_pos[0] += 1
                posx.append(self.current_pos[0])
                posy.append(self.current_pos[1])
                if np.abs(self.current_pos[0]) == self.current_layer:
                    self.status = 2
            elif self.status == 2:
                self.current_pos[1] -= 1
                posx.append(self.current_pos[0])
                posy.append(self.current_pos[1])
                if np.abs(self.current_pos[1]) == self.current_layer:
                    self.status = 3
            elif self.status == 3:
                self.current_pos[0] -= 1
                posx.append(self.current_pos[0])
                posy.append(self.current_pos[1])
                if np.abs(self.current_pos[0]) == self.current_layer:
                    self.status = 0

        return posx, posy

    def show_last_layer(self):
        print(self.last_layer)

    def main(self):
        poslist = []
        while True:
            #self.show_current_position()
            result = self.get_next()
            if result == 1:
                break


if __name__ == '__main__':
    unicarsel_line = UnicarsalLine(last_layer=int(sys.argv[1]))
    posx, posy = unicarsel_line.get_list()
    for x, y in zip(posx, posy):
        print("%f,%f" % (x, y))
    #unicarsel_line.main()

