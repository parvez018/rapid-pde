# Efficient Learning of Sparse and Decomposable PDEs using Random Projection
Code repository for RAPID-PDE algorithm paper accepted in UAI-2022 for publication
## Abstract
Learning physics models in the form of Partial Differential Equations (PDEs) is carried out through back-propagation to match the simulations of the physics model with experimental observations. Nevertheless, such matching involves computation over billions of elements, presenting a significant computational overhead. We notice many PDEs in real world problems are sparse and decomposable, where the temporal updates and the spatial features are sparsely concentrated on small interface regions. We propose Rapid-PDE, an algorithm to expedite the learning of sparse and decomposable PDEs. Our Rapid-PDE first uses random projection to compress the high dimensional sparse updates and features into low dimensional representations and then use these compressed signals during learning. Crucially, such a conversion is only carried out once prior to learning and the entire learning process is conducted in the compressed space. Theoretically, we derive a constant factor approximation between the projected loss function and the original one with logarithmic number of projected dimensions. Empirically, we demonstrate Rapid-PDE with data compressed to 0.05% of its original size learns similar models compared with uncompressed algorithms in learning a set of phase-field models which govern the spatial-temporal dynamics of nano-scale structures in metallic materials.

## Full Paper and Presentation Video Link
[Download the paper here](https://openreview.net/pdf?id=SCg9JDUscgq)

[Video presentation link to be added later]()


## Reference

```
@inproceedings{nasim2022efficient,
  title={Efficient Learning of Sparse and Decomposable PDEs using Random Projection},
  author={Nasim, Md and Zhang, Xinghang and El-Azab, Anter and Xue, Yexiang},
  booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
  year={2022}
}

```


