"""Nothing to configure: every test reads `data/*.json` through
`drainlag.data.load()`, which walks up to the directory holding
`data/sweep.json`. The tests read exactly the files a reader would.
"""
