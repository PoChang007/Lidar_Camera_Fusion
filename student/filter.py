# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # return system matrix F
        ############

        dt = params.dt
        return np.matrix([[1, 0, 0, dt, 0, 0],
                          [0, 1, 0, 0, dt, 0],
                          [0, 0, 1, 0, 0, dt],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])

        ############
        # END student code
        ############

    def Q(self):
        ############
        # return process noise covariance Q
        ############

        dt = params.dt
        q = params.q
        q1 = ((dt**3)/3) * q
        q2 = ((dt**2)/2) * q
        q3 = dt * q

        return np.matrix([[q1, 0, 0, q2, 0, 0],
                          [0, q1, 0, 0, q2, 0],
                          [0, 0, q1, 0, 0, q2],
                          [q2, 0, 0, q3, 0, 0],
                          [0, q2, 0, 0, q3, 0],
                          [0, 0, q2, 0, 0, q3]])

        ############
        # END student code
        ############

    def predict(self, track):
        ############
        # predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        x = track.x
        P = track.P
        F = self.F()
        Q = self.Q()

        x = F*x
        P = F*P*F.transpose() + Q

        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############

    def update(self, track, meas):
        ############
        # update state x and covariance P with associated measurement, save x and P in track
        ############

        x = track.x
        P = track.P
        H = meas.sensor.get_H(x)

        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)
        K = P*H.transpose()*np.linalg.inv(S)
        x = x + K*gamma
        I = np.identity(params.dim_state)
        P = (I - K*H) * P

        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # calculate and return residual gamma
        ############

        x = track.x
        hx = meas.sensor.get_hx(x)
        z = meas.z
        gamma = z - hx

        return gamma

        ############
        # END student code
        ############

    def S(self, track, meas, H):
        ############
        # calculate and return covariance of residual S
        ############

        P = track.P
        R = meas.R
        S = H*P*H.transpose() + R

        return S

        ############
        # END student code
        ############
