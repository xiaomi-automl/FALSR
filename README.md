# Fast, Accurate and Lightweight Super-Resolution models [![arXiv](http://img.shields.io/badge/cs.CV-arXiv%3A1901.07261-B31B1B.svg)](https://arxiv.org/abs/1901.07261)

We present FALSR A,B,C models. The metrics and results can be generated with,

```shell
$ python3 calculate.py --pb_path ./pretrained_model/FALSR-A.pb --save_path ./result/
```

## Comparison with state-of-the-art methods

| Method | MulAdds| Params |Set5 | Set14 | BSD100 | Urban100 | 
| ----------------- | ------------- |------------- |------------- |------------- |-------------|------------- |
| SRCNN  | 52.7G | 57K | 36.66/0.9542 | 32.42/0.9063 | 31.36/0.8879 | 29.50/0.8946 |
| FSRCNN | 6.0G | 12K | 37.00/0.9558 | 32.63/0.9088 | 31.53/0.8920 | 29.88/0.9020 |
| VDSR  | 612.6G | 665K | 37.53/0.9587 | 33.03/0.9124 | 31.90/0.8960 | 30.76/0.9140 |
| DRCN   | 17,974.3G | 1,774K | 37.63/0.9588| 33.04/0.9118| 31.85/0.8942| 30.75/0.9133 |
| CNF  | 311.0G | 337K | 37.66/0.9590 | 33.38/0.9136 | 31.91/0.8962 | -|
| LapSRN  | 29.9G | 813K | 37.52/0.9590| 33.08/0.9130| 31.80/0.8950 | 30.41/0.9100 |
| DRRN  | 6,796.9G | 297K | 37.74/0.9591 | 33.23/0.9136 | 32.05/0.8973 | 31.23/0.9188 |
| BTSRN  | 207.7G | 410K | 37.75/- | 33.20/- | 32.05/- | 31.63/-	|
| MemNet  | 2,662.4G | 677K | 37.78/0.9597 | 33.28/0.9142 | 32.08/0.8978 | 31.31/0.9195 |
| SelNet  | 225.7G | 974K | 37.89/0.9598 | 33.61/0.9160 | 32.08/0.8984 | -|
| CARN  | 222.8G | 1,592K | 37.76/0.9590 | 33.52/0.9166| 32.09/0.8978 | 31.92/0.9256|
| CARN-M  | 91.2G | 412K | 37.53/0.9583 | 33.26/0.9141 | 31.92/0.8960 | 31.23/0.9194|
| MoreMNAS-A   | 238.6G | 1039K | 37.63/0.9584 | 33.23/0.9138 | 31.95/0.8961 | 31.24/0.9187|
| MoreMNAS-B  | 256.9G | 1118K | 37.58/0.9584 | 33.22/0.9135 | 31.91/0.8959| 31.14/0.9175|
| MoreMNAS-C  | 5.5G | 25K | 37.06/0.9561 | 32.75/0.9094| 31.50/0.8904 | 29.92/0.9023|
| MoreMNAS-D  | 152.4G | 664K | 37.57/0.9584 | 33.25/0.9142 | 31.94/0.8966 | 31.25/0.9191|
| FALSR-A (ours) |234.7G | 1,021K | 37.82/0.9595 | 33.55/0.9168	| 32.12/0.8987 | 31.93/0.9256|
| FALSR-B (ours) | 74.7G | 326k | 37.61/0.9585	| 33.29/0.9143 | 31.97/0.8967 	| 31.28/0.9191 |
| FALSR-C (ours) | 93.7G |408k | 37.66/0.9586	| 33.26/0.9140 | 31.96/0.8965	| 31.24/0.9187 |


## Citation

Your citations are welcomed!

    @inproceedings{chu2020fast,
      title={Fast, accurate and lightweight super-resolution with neural architecture search},
      author={Chu, Xiangxiang and Zhang, Bo and Ma, Hailong and Xu, Ruijun and Li, Qingyuan},
      booktitle={International Conference on Pattern Recognition, ICPR},
      year={2020}
    }
