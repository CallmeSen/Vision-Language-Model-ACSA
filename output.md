torchrun --standalone --nproc_per_node=1 training.py
Device: cuda
Ensuring base models are cached...
  Caching LLM: Qwen/Qwen3-4B-Instruct-2507
Fetching 13 files: 100%|████████████████████████████████████████████| 13/13 [00:00<00:00, 51978.98it/s]
Download complete: : 0.00B [00:00, ?B/s]                Caching SigLIP: google/siglip2-so400m-patch16-256
Fetching 9 files: 100%|████████████████████████████████████████████████| 9/9 [00:00<00:00, 5718.64it/s]
Downloading (incomplete total...): 0.00B [00:00, ?B/s]All base models cached successfully.
Download complete: : 0.00B [00:00, ?B/s]                                         | 0/9 [00:00<?, ?it/s]
Download complete: : 0.00B [00:01, ?B/s]

============================================================
Training MultimodalACSAModel (all 6 aspects, encode-once, aspect-loop)
============================================================
Initializing model (downloading base models if needed)...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████| 398/398 [00:00<00:00, 9418.22it/s]
Loading weights: 100%|█████████████████████████████████████████████| 888/888 [00:00<00:00, 8060.17it/s]
Trainable params: {'total': 216724997, 'lora': 23592960, 'other': 193132037}

Epoch 1/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:54<00:00,  1.02it/s, loss=1.1046]
Train Loss: 1.1046
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:16<00:00,  1.64it/s]
Dev Loss: 0.8810
Dev F1 (macro): 0.4021  Precision: 0.4778  Recall: 0.4004  Accuracy: 0.6602
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.2228  P=0.3385  R=0.2598  Acc=0.6870
  Public_area: F1=0.3443  P=0.3138  R=0.4092  Acc=0.6230
  Location: F1=0.2100  P=0.1810  R=0.2500  Acc=0.7240
  Food: F1=0.2169  P=0.1915  R=0.2500  Acc=0.7660
  Room: F1=0.3911  P=0.4655  R=0.3848  Acc=0.5600
  Service: F1=0.2935  P=0.3250  R=0.2981  Acc=0.6010
*** New best F1: 0.4021 (P=0.4778, R=0.4004) ***

Epoch 2/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:47<00:00,  1.04it/s, loss=0.8841]
Train Loss: 0.8841
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:17<00:00,  1.62it/s]
Dev Loss: 0.7655
Dev F1 (macro): 0.4952  Precision: 0.6425  Recall: 0.4705  Accuracy: 0.6933
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.3821  P=0.3852  R=0.4122  Acc=0.7490
  Public_area: F1=0.4777  P=0.5251  R=0.4733  Acc=0.7380
  Location: F1=0.2952  P=0.3151  R=0.2968  Acc=0.7280
  Food: F1=0.3106  P=0.3012  R=0.3840  Acc=0.6760
  Room: F1=0.5530  P=0.6075  R=0.5458  Acc=0.6190
  Service: F1=0.2898  P=0.4731  R=0.3013  Acc=0.6500
*** New best F1: 0.4952 (P=0.6425, R=0.4705) ***

Epoch 3/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:40<00:00,  1.06it/s, loss=0.7052]
Train Loss: 0.7052
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:20<00:00,  1.55it/s]
Dev Loss: 0.6202
Dev F1 (macro): 0.6120  Precision: 0.6076  Recall: 0.6253  Accuracy: 0.7618
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.4948  P=0.5869  R=0.4739  Acc=0.7790
  Public_area: F1=0.5444  P=0.5868  R=0.5841  Acc=0.6790
  Location: F1=0.3896  P=0.3822  R=0.3974  Acc=0.8180
  Food: F1=0.3680  P=0.5343  R=0.3425  Acc=0.8070
  Room: F1=0.5687  P=0.6053  R=0.6039  Acc=0.6570
  Service: F1=0.5516  P=0.5259  R=0.6180  Acc=0.8310
*** New best F1: 0.6120 (P=0.6076, R=0.6253) ***

Epoch 4/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:42<00:00,  1.05it/s, loss=0.5620]
Train Loss: 0.5620
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:14<00:00,  1.67it/s]
Dev Loss: 0.4998
Dev F1 (macro): 0.5670  Precision: 0.7168  Recall: 0.5405  Accuracy: 0.8112
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.4208  P=0.5370  R=0.4209  Acc=0.7920
  Public_area: F1=0.5196  P=0.7142  R=0.5139  Acc=0.7650
  Location: F1=0.4095  P=0.4236  R=0.4039  Acc=0.8580
  Food: F1=0.5180  P=0.7657  R=0.4967  Acc=0.8680
  Room: F1=0.5680  P=0.6948  R=0.5318  Acc=0.7060
  Service: F1=0.5134  P=0.5938  R=0.5029  Acc=0.8780
No improvement. Patience: 1/4

Epoch 5/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:46<00:00,  1.04it/s, loss=0.4677]
Train Loss: 0.4677
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:13<00:00,  1.70it/s]
Dev Loss: 0.4853
Dev F1 (macro): 0.6320  Precision: 0.6899  Recall: 0.6050  Accuracy: 0.8265
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.5367  P=0.6363  R=0.5085  Acc=0.8180
  Public_area: F1=0.5922  P=0.7038  R=0.5724  Acc=0.7850
  Location: F1=0.3964  P=0.4243  R=0.3866  Acc=0.8440
  Food: F1=0.5284  P=0.6969  R=0.4998  Acc=0.8680
  Room: F1=0.6473  P=0.6712  R=0.6382  Acc=0.7670
  Service: F1=0.5786  P=0.5715  R=0.5911  Acc=0.8770
*** New best F1: 0.6320 (P=0.6899, R=0.6050) ***

Epoch 6/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:47<00:00,  1.04it/s, loss=0.4012]
Train Loss: 0.4012
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:15<00:00,  1.66it/s]
Dev Loss: 0.4715
Dev F1 (macro): 0.6489  Precision: 0.6812  Recall: 0.6332  Accuracy: 0.8322
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.5190  P=0.5729  R=0.5028  Acc=0.8030
  Public_area: F1=0.5981  P=0.6187  R=0.5907  Acc=0.7770
  Location: F1=0.4612  P=0.5147  R=0.4650  Acc=0.8730
  Food: F1=0.6066  P=0.6619  R=0.5768  Acc=0.8820
  Room: F1=0.6794  P=0.7063  R=0.6668  Acc=0.7790
  Service: F1=0.5891  P=0.5806  R=0.6021  Acc=0.8790
*** New best F1: 0.6489 (P=0.6812, R=0.6332) ***

Epoch 7/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:44<00:00,  1.05it/s, loss=0.3211]
Train Loss: 0.3211
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:16<00:00,  1.63it/s]
Dev Loss: 0.5000
Dev F1 (macro): 0.6507  Precision: 0.6948  Recall: 0.6263  Accuracy: 0.8340
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.5468  P=0.6130  R=0.5250  Acc=0.8120
  Public_area: F1=0.5784  P=0.6275  R=0.5674  Acc=0.7730
  Location: F1=0.4957  P=0.5902  R=0.4818  Acc=0.8830
  Food: F1=0.5664  P=0.6134  R=0.5448  Acc=0.8730
  Room: F1=0.6922  P=0.7121  R=0.6778  Acc=0.7820
  Service: F1=0.5913  P=0.5887  R=0.5943  Acc=0.8810
*** New best F1: 0.6507 (P=0.6948, R=0.6263) ***

Epoch 8/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:44<00:00,  1.05it/s, loss=0.2516]
Train Loss: 0.2516
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:16<00:00,  1.63it/s]
Dev Loss: 0.6063
Dev F1 (macro): 0.6488  Precision: 0.6739  Recall: 0.6345  Accuracy: 0.8095
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.5170  P=0.5680  R=0.5134  Acc=0.7680
  Public_area: F1=0.5914  P=0.6280  R=0.5994  Acc=0.7370
  Location: F1=0.5072  P=0.5312  R=0.5083  Acc=0.8420
  Food: F1=0.5604  P=0.6166  R=0.5445  Acc=0.8770
  Room: F1=0.6325  P=0.6951  R=0.6080  Acc=0.7410
  Service: F1=0.5900  P=0.6285  R=0.5725  Acc=0.8920
No improvement. Patience: 1/4

Epoch 9/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:54<00:00,  1.02it/s, loss=0.1839]
Train Loss: 0.1839
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:20<00:00,  1.56it/s]
Dev Loss: 0.6539
Dev F1 (macro): 0.6474  Precision: 0.6899  Recall: 0.6225  Accuracy: 0.8298
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.5076  P=0.6358  R=0.4810  Acc=0.8070
  Public_area: F1=0.5820  P=0.6052  R=0.5760  Acc=0.7510
  Location: F1=0.4612  P=0.5405  R=0.4610  Acc=0.8720
  Food: F1=0.5281  P=0.6082  R=0.5192  Acc=0.8750
  Room: F1=0.6884  P=0.7005  R=0.6853  Acc=0.7840
  Service: F1=0.5809  P=0.6069  R=0.5689  Acc=0.8900
No improvement. Patience: 2/4

Epoch 10/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:55<00:00,  1.01it/s, loss=0.1395]
Train Loss: 0.1395
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:15<00:00,  1.67it/s]
Dev Loss: 0.7018
Dev F1 (macro): 0.6484  Precision: 0.6852  Recall: 0.6289  Accuracy: 0.8255
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.5419  P=0.5790  R=0.5241  Acc=0.8030
  Public_area: F1=0.5911  P=0.6254  R=0.5900  Acc=0.7530
  Location: F1=0.5530  P=0.6631  R=0.5221  Acc=0.8840
  Food: F1=0.5582  P=0.6077  R=0.5385  Acc=0.8730
  Room: F1=0.6751  P=0.7046  R=0.6579  Acc=0.7650
  Service: F1=0.5816  P=0.6396  R=0.5593  Acc=0.8750
No improvement. Patience: 3/4

Epoch 11/15
Training: 100%|█████████████████████████████████████████| 360/360 [05:50<00:00,  1.03it/s, loss=0.1052]
Train Loss: 0.1052
Evaluating: 100%|████████████████████████████████████████████████████| 125/125 [01:17<00:00,  1.61it/s]
Dev Loss: 0.7702
Dev F1 (macro): 0.6491  Precision: 0.6913  Recall: 0.6245  Accuracy: 0.8277
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.5430  P=0.6037  R=0.5190  Acc=0.8050
  Public_area: F1=0.5870  P=0.6587  R=0.5705  Acc=0.7600
  Location: F1=0.5199  P=0.6101  R=0.4901  Acc=0.8770
  Food: F1=0.5294  P=0.6240  R=0.5051  Acc=0.8660
  Room: F1=0.6816  P=0.6930  R=0.6742  Acc=0.7710
  Service: F1=0.5956  P=0.6178  R=0.5802  Acc=0.8870
No improvement. Patience: 4/4
Early stopping at epoch 11

Best overall F1: 0.6507
Per-epoch results saved to outputs/train_result.json
Best dev results saved to outputs/dev_result.json

============================================================
Loading best model for test evaluation...
============================================================
Loading weights: 100%|█████████████████████████████████████████████| 398/398 [00:00<00:00, 9599.27it/s]
Loading weights: 100%|█████████████████████████████████████████████| 888/888 [00:00<00:00, 9076.35it/s]
Loaded best checkpoint from outputs/best_checkpoint.pt
Test Evaluation: 100%|███████████████████████████████████████████████| 125/125 [01:14<00:00,  1.68it/s]

Test Loss: 0.5106
Test F1 (macro): 0.6538  Precision: 0.6893  Recall: 0.6316  Accuracy: 0.8262
Per-aspect metrics (F1 / Precision / Recall / Acc):
  Facilities: F1=0.5457  P=0.5644  R=0.5309  Acc=0.8040
  Public_area: F1=0.5456  P=0.5880  R=0.5421  Acc=0.7700
  Location: F1=0.4426  P=0.4729  R=0.4419  Acc=0.8750
  Food: F1=0.5911  P=0.6327  R=0.5634  Acc=0.8590
  Room: F1=0.7114  P=0.7205  R=0.7041  Acc=0.7830
  Service: F1=0.5933  P=0.5998  R=0.5919  Acc=0.8660