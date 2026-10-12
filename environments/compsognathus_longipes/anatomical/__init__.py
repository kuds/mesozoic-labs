"""Gymnasium training environments for the anatomical and robot models."""

from .compsognathus_env import (
    CompsognathusLongipesAnatomicalEnv,
    CompsognathusLongipesEnv,
    CompsognathusLongipesRobotEnv,
)

__all__ = ["CompsognathusLongipesAnatomicalEnv", "CompsognathusLongipesEnv", "CompsognathusLongipesRobotEnv"]
