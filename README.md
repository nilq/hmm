# Hidden Markov Model of Visual Attention 

## Getting started

### Environment 🌍

To save the environment, we use Conda. Simply create and activate the environment in the following way:

```
conda env create -f conda.yaml
conda env activate hmm
```

### Dependencies 📦

We use Poetry (v1.3.1) to handle dependencies within the environment. Simply install dependencies like this:

```
python -m poetry install
```

You can use your own Poetry if you feel like it. No worries.


### Linting ✨

University doesn't pay us enough to test things. However, the code should still be comfortable.
We use [Ruff](https://ruff.rs) to make sure things are nice.

Run the following before pushing, please:

```
ruff check hmm/
poetry run black .
poetry run isort .
```

These things can be automated with IDE extensions.

## License

> Can you even license university course work?

> Yes, with the Un'i'license! "bu dum tsh"

The Unlicense. 🤓
