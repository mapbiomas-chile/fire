import os

models = {
        "r1"  : "col1_chile_v1_r1_rnn_lstm_ckpt",
        "r2"  : "col1_chile_v1_r2_rnn_lstm_ckpt",
        "r4"  : "col1_chile_v2_r4_rnn_lstm_ckpt",
        "r6"  : "col1_chile_v2_r6_rnn_lstm_ckpt"
}
start_year = 2013
end_year = 2025

for region,model in models.items():
    for year in range(start_year, end_year + 1):
        tile = f"b14_chile_{region}_{year}_cog.tif"
        command = f"sbatch run_classify_fire_model_slurm.sh {model} {tile}"
        print(command)
        os.system(command)