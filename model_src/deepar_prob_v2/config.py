config = {
    "features": [
        {
            "name": "链条震动",
            "type": "target",
            "categorical": False,
        },
        {
            "name": "电机震动",
            "type": "target",
            "categorical": False,
        },
        {
            "name": "电机温度",
            "type": "target",
            "categorical": False,

        },
        {
            "name": "A相电流",
            "type": "covariate",
            "categorical": False,
        },
        {
            "name": "B相电流",
            "type": "covariate",
            "categorical": False,
        },
        {
            "name": "C相电流",
            "type": "covariate",
            "categorical": False,
        },
        {
            "name": "R-位置值",
            "type": "covariate",
            "categorical": True,
            "n_values": 8,
            "embedding_size": 2
        },
        {
            "name": "day_of_week",
            "type": "covariate",
            "categorical": True,
            "n_values": 7,
            "embedding_size": 2
        },
        {
            "name": "hour_of_day",
            "type": "covariate",
            "categorical": True,
            "n_values": 24,
            "embedding_size": 3
        },
    ],
    "model_architecture": {
        # "encoder": {
        #     "type": "LSTMEncoder",
        #     "n_target": 6,
        #     "layers": [
        #         (72, "relu")
        #     ],
        # },
        "encoder":{
            "type": "ConvEncoder",
            "n_target": 6,
            "series_len": 4,
            "layers": [
                (72, 3, "relu"),
            ],
        },
        "decoder": {
            "input_dim": 72,
            "n_target": 6,
            "type": "DecoderDiag",
        },
        "prob_layer": "NegLogLikelihood",
        "metrics": ["MAE", "MAPE"],
    },
    "ckpt_path": "./ckpt/",
    "save_path": "./model_binary/",
}
