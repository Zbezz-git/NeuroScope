"""
Command-line interface for NeuroScope.

Provides utilities for running the server standalone and
testing model visualization.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="neuroscope",
        description="Real-time neural network observability tool",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the visualization server")
    server_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)",
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address to bind to (default: localhost)",
    )
    server_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo with a sample model")
    demo_parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn", "transformer"],
        default="mlp",
        help="Demo model type (default: mlp)",
    )

    # Version command
    subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    if args.command == "server":
        return run_server(args)
    elif args.command == "demo":
        return run_demo(args)
    elif args.command == "version":
        return show_version()
    else:
        parser.print_help()
        return 0


def run_server(args) -> int:
    """Run the NeuroScope server."""
    from neuroscope.core.server import NeuroScopeServer

    print("[NeuroScope] Starting server...")
    print(f"[NeuroScope] Host: {args.host}:{args.port}")

    server = NeuroScopeServer(host=args.host, port=args.port)
    server.start(open_browser=not args.no_browser)

    print("[NeuroScope] Server running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[NeuroScope] Shutting down...")
        server.stop()

    return 0


def run_demo(args) -> int:
    """Run a demo with a sample model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[ERROR] PyTorch is required for demo. Install with: pip install torch")
        return 1

    import neuroscope

    print("[NeuroScope] Demo")
    print(f"[NeuroScope] Creating {args.model} model...")

    # Create demo model
    if args.model == "mlp":
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        sample_input = torch.randn(32, 784)
    elif args.model == "cnn":
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 10),
        )
        sample_input = torch.randn(4, 3, 32, 32)
    else:  # transformer
        model = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=2048,
            batch_first=True,
        )
        sample_input = (torch.randn(2, 10, 512), torch.randn(2, 10, 512))

    # Attach and run
    print("[NeuroScope] Attaching tracer...")
    neuroscope.attach(model)
    neuroscope.start_server()

    print("[NeuroScope] Running forward pass...")
    with torch.no_grad():
        if isinstance(sample_input, tuple):
            _ = model(*sample_input)
        else:
            _ = model(sample_input)

    if neuroscope._active_tracer:
        graph = neuroscope._active_tracer.get_graph()
        print(f"[NeuroScope] Captured {len(graph)} nodes")

    print("\n[NeuroScope] View graph in browser. Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[NeuroScope] Cleaning up...")
        neuroscope.detach()
        neuroscope.stop_server()

    return 0


def show_version() -> int:
    """Show version information."""
    from neuroscope import __version__
    print(f"NeuroScope {__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
