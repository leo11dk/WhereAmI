"""Common lifecycle for standalone ROS nodes launched by the official runner."""

import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions


def run_node(factory, args=None):
    # Keep the context alive until callbacks stop. Humble's asynchronous signal
    # shutdown can otherwise invalidate a subscription while taking a message.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    previous = {sig: signal.signal(sig, signal.default_int_handler)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    node = None
    try:
        node = factory()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # main.py signals the process group, then ROS launch forwards SIGINT.
        # A second interrupt must not abort destruction halfway through.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if node is not None:
                close = getattr(node, 'close', None)
                if close is not None and node.context.ok():
                    close()
                node.destroy_node()
        finally:
            try:
                rclpy.try_shutdown()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
