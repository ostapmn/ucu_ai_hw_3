# Explainability summary

Linear model: top tokens per class in `explain_linear_top_tokens.csv` and `images/xai_linear_top_tokens.png`.

Transformer: integrated gradients on head+tail windows of English validation texts, 8 per true class.

Mean absolute attribution per token, by predicted class and window part:

```
part          head     tail
predicted                  
ai         0.00001  0.00001
fusion     0.00001  0.00001
human      0.00002  0.00001
```

Share of total absolute attribution by true class and window part:

```
part     head   tail
label               
ai      0.437  0.563
fusion  0.326  0.674
human   0.310  0.690
```
