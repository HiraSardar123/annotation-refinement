"""SAM2 refinement pipeline for the welding annotation masks.

One module per step:

    segregate.py    step 1   split the human mask into molten / droplet components
    boxes.py        step 2   one padded box per component (molten 4 px, droplet 1 px)
    prompts.py      step 3   2 positive + 3 negative points per component
    sam_predict.py  step 4   SAM2 predictions (and per-class SAM probabilities)
    relabel.py      step 5   label pixels the refinement left blank
    instances.py             one solid mask per molten / droplet instance (after 5, 7, 9)
    molten_top.py            only molten / background above the molten (after 5, 7, 9)
    gaps.py         step 7   fill gaps between annotated classes
    spatter.py      step 9   remove static (noise) spatter
    visualize.py             box / point debug images and human|SAM comparisons
    evaluate.py     step 11  per-image, per-class pixel counts for every stage
    main.py         step 12  runs everything; see run_dataset.py / run_folder.py
"""
