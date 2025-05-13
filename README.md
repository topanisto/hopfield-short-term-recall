# working-memory
For 9.530's Final Project, by Anka Hu.

## Overview

Hopfield networks are compelling solutions to the biological memory question, which asks one to retrieve a stored prototype vector, or "memory", upon given a noisy input, but how well do they answer to the phenomena of short-term memory? In particular, can the behaviors of storing and quickly misremembering new information, such as forgetting someone's name after meeting them, or deja vu for a word you've never seen before, emerge from Hopfield networks?

This project explores the following:
- A candidate algorithm for short-term storing and recall mechanisms in Hopfield networks.
- A verbal memory task, where participants are asked to mark whether or not they've seen strings of characters, and analysis of empirical data collected.
- An empirical exploration of the number of prototypes stored by a Hopfield network and the probability of the network converging to a prototype state.

## How this repo is organized
The bulk of the work is recoreded in the `main.ipynb` notebook, which describes the algorithm, task, and data analysis. `hopfield.py` contains our implementation for a discrete Hopfield Network implementing an asynchronous update rule. `wm_single.html` is the raw source code for the short-term verbal memory cognitive task, implemented in jsPsych and playable either at [cognition.run](https://wea3utkzqr.cognition.run) or in-browser.
