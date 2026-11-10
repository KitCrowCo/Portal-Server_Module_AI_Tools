"""Asio - simulates AI work on a pool of machines: what fits where, how long it takes, what holds it back, what it costs, how sure that is, and which setup is best for a goal.
values: tiered numbers and cases. catalog: swappable data. pool: machines and links. perf: step physics. sim: discrete events. workloads: kinds of work. report: metrics and bottlenecks. run: cases, Monte Carlo, sweeps, goal seeking (with Athene). viz: SVG charts."""
from asio_core.catalog import Catalog
from asio_core.values import Valuer, CASES, TIERS, describe
from asio_core.run import once, cases, monte_carlo, sweep, optimize

__version__ = "0.1.0"
