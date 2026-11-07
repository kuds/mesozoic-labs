"""Onboard RGB-D person selection and velocity commands."""

from .controller import Command, OnboardFollower
from .perception import CameraGeometry, RGBDFrame
from .web import OperatorServer

__all__ = ["CameraGeometry", "Command", "OnboardFollower", "OperatorServer", "RGBDFrame"]
