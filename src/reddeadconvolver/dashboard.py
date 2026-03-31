from __future__ import annotations


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Uvicorn is required to run the dashboard. Install with: pip install 'reddeadconvolver[web]'"
        ) from exc

    uvicorn.run("reddeadconvolver.webapp:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
