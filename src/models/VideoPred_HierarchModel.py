"""
Abstract base class for Hierarchical Stochastic Video Prediction models.
"""

from copy import deepcopy
from collections import defaultdict
import torch
import models


class HierarchModel(models.VideoPredModel):
    """ Base class for hierarchical video prediction models """

    def __init__(self, model_params, linear=True, stochastic=False, use_output=True, **kwargs):
        """ Hierarchical model intializer """
        super().__init__(model_params=model_params, linear=linear, **kwargs)
        self.stochastic = stochastic # stochastic = True
        self.use_output = use_output # use_output =True
        self.hidden_size = self.model_params["HierarchLSTM"]["hidden_dim"] # hidden_size = 128
        self.num_hierarch = self.model_params["HierarchLSTM"]["num_hierarch"] # num_hierarch = 3
        self.num_lstm_layers = self.model_params["HierarchLSTM"]["num_layers"] # 4
        if not isinstance(self.num_lstm_layers, list): # True
            self.num_lstm_layers = [self.num_lstm_layers] * self.num_hierarch # [4,4,4]
        assert len(self.num_lstm_layers) == self.num_hierarch

        self.periods = self.model_params["HierarchLSTM"]["periods"] # periods = [1,4,8]
        assert isinstance(self.periods, list) and (len(self.periods) == self.num_hierarch)
        self.ancestral_sampling = self.model_params["HierarchLSTM"]["ancestral_sampling"] # ancestral_sampling = True
        self.autoreg_mode = self.model_params["autoreg_mode"] # autoreg_mode = LAST_PRED_FEATS
        assert self.autoreg_mode in ["LAST_PRED_FRAME", "LAST_PRED_FEATS"]
        self.last_context_residuals = self.model_params["last_context_residuals"] # last_context_residuals = True

        # instanciating modules
        self.posterior = self._get_prior_post(model_key="LSTM_Posterior") if self.stochastic else None
        self.prior = self._get_prior_post(model_key="LSTM_Prior") if self.stochastic else None
        self.predictor = self._get_predictor()
        self.aux_outputs = self.model_params["HierarchLSTM"]["aux_outputs"]
        if self.aux_outputs: # True
            self.n_hmap_channels = self.model_params["HierarchLSTM"]["n_hmap_channels"] # n_hmap_channels = [1,1]
            self.decoder_heads = self._get_decoder_heads()
        return

    def forward(self, x, context=4, num_preds=10, teacher_force=False, openloop=False):
        """ Forward pass through Hierarchical models. """ # x = 16 * 57 * 3 * 64 * 64,context = 17, num_preds = [5,5,5],
        batch_size, seq_len = x.shape[:2]  # batch_size = 16, seq_len = 57
        if not isinstance(num_preds, list):
            num_preds = [num_preds] * self.num_hierarch
        assert seq_len >= context + num_preds[0]

        # initializing LSTM counters and outputs structure
        self.init_hidden(batch_size=batch_size) # batch_size = 16
        self.init_counter()
        empty_lists = [[] for _ in range(self.num_hierarch)] # num_hierarch = 3
        out_dict = {"preds": {}, "target_masks": defaultdict(lambda: torch.full((seq_len,), False)),
                    "mu_post": deepcopy(empty_lists), "logvar_post": deepcopy(empty_lists),
                    "mu_prior": deepcopy(empty_lists), "logvar_prior": deepcopy(empty_lists),
                    "latents": deepcopy(empty_lists)}
        inputs = targets = x.float()
        preds = defaultdict(list)
        feats_dict = defaultdict(list)
        pred_feats = None

        # autoregressive prediction loop
        for t in range(0, seq_len-1): # seq_len = 57,(0,1,2,3,...55)
            if self.stochastic: # True
                # get target features for computing approx. posterior
                target_feats = self.encoder(targets[:, t+1]) # [:,1],[:,2],...[:,56] / target_feats = 16 * 512 * 4 * 4,[16 * 64 * 32 * 32,16 * 128 * 16 * 16, 16 * 256 * 8 * 8]
                feats_dict["target"] = self._get_input_feats(*target_feats)
                # feats_dict["target"]  = [16 * 128 * 16 * 16, 16 * 256 * 8 * 8,16 * 512 * 4 * 4]
            # get current input features
            if (t < context):# t < 17
                enc_feats = self.encoder(inputs[:, t])
                feats_dict["input"] = self._get_input_feats(*enc_feats) # feats_dict["input"]  = [16 * 128 * 16 * 16, 16 * 256 * 8 * 8,16 * 512 * 4 * 4]
                feats_dict["residuals"] = self._get_residual_feats(*enc_feats) # feats_dict["residuals"] = [16 * 64 * 32 * 32, 16 * 128 * 16 * 16, 16 * 256 * 8 * 8, 16 * 512 * 4 * 4]
            else:
                if self.autoreg_mode == "LAST_PRED_FEATS":
                    feats_dict["input"] = pred_feats
                else:  # re-encoding previous image
                    assert self.autoreg_mode == "LAST_PRED_FRAME"
                    enc_feats = self.encoder(inputs[:, t]) if teacher_force else self.encoder(preds[0][-1])
                    feats_dict["input"] = self._get_input_feats(*enc_feats)

            # feeding features to hierarchical predictor in order to forecast next features # 将特征馈送到分层预测器以预测下一个特征
            pred_feats, out_dict = self.predict(
                    feats_dict=feats_dict,
                    out_dict=out_dict,
                    cur_frame=t,
                    context=context
                ) # pred_feats = [16 * 128 * 16 * 16,16 * 256 * 8 * 8,16 * 512 * 4 * 4]

            # decoding predicted frames # 解码预测帧
            dec_inputs = self._get_decoder_inputs(pred_feats, feats_dict["residuals"]) # [16 * 512 * 4 * 4,[16 * 64 * 32 * 32, 16 * 128 * 16 * 16, 16 * 256 * 8 * 8]]
            pred_output, dec_skips = self.decoder(dec_inputs)#  pred_output = 16 * 3 * 64 * 64, dec_skips =[16 * 256 * 8 * 8,16 * 128 * 16 * 16,16 * 64 * 32 * 32]
            if (t >= context-1): # t >= 17 - 1 = 16
                if (torch.count_nonzero(out_dict["target_masks"][0]) < num_preds[0]):
                    preds[0].append(pred_output)
                    out_dict["target_masks"][0][t+1] = True
                elif openloop:
                    preds[0].append(pred_output)

            # decoding high-level predictions, if available #解码高级预测（如果有）
            if self.aux_outputs: # True
                dec_head_inputs = self._get_decoder_head_inputs(pred_feats, dec_skips) # dec_head_inputs = [16 * 128 * 16 * 16,16 * 256 * 8 * 8]
                for h in range(1, self.num_hierarch):# num_hierarch = 3, (1,2)
                    if (t >= context-1 and (len(preds[h]) < num_preds[h] or openloop)): # t >= 17-1 = 16 and (0 < 5 or False) = False
                        pred_out, ticked = self.decoder_heads[h-1](dec_head_inputs[h-1]) # 16 * 1 * 64 * 64
                        if ticked and (torch.count_nonzero(out_dict["target_masks"][h]) < num_preds[h]):
                            preds[h].append(pred_out)
                            p = self.decoder_heads[h-1].period
                            out_dict["target_masks"][h][t+p] = True
                        elif openloop:
                            preds[h].append(pred_out)

        # reshaping of tensors
        if self.stochastic: # True
            for i in range(self.num_hierarch): # num_hierarch = 3,[1,2,3]
                for k in ["mu_post", "logvar_post", "mu_prior", "logvar_prior"]:
                    out_dict[k][i] = out_dict[k][i][:len(preds[i])]
        for h, pred_list in preds.items():
            out_dict["preds"][h] = torch.stack(pred_list, dim=1)
        if not self.aux_outputs:
            out_dict["preds"] = out_dict["preds"][0]
            out_dict["target_masks"] = out_dict["target_masks"][0]
        return out_dict

    def _get_input_feats(self, enc_outs, enc_skips):
        raise NotImplementedError("Abstract class does not implement '_get_input_feats'")

    def _get_residual_feats(self, enc_outs, enc_skips):
        raise NotImplementedError("Abstract class does not implement '_get_residual_feats'")

    def _get_decoder_head_inputs(self, pred_feats, dec_skips):
        raise NotImplementedError("Abstract class does not implement '_get_decoder_head_inputs'")

    def _get_decoder_inputs(self, pred_feats, residuals):
        raise NotImplementedError("Abstract class does not implement '_get_decoder_inputs'")

    def _get_decoder_heads(self):
        raise NotImplementedError("Abstract class does not implement '_get_decoder_heads'")

    def init_counter(self):
        super().init_counter()
        if self.aux_outputs:
            for dec_head in self.decoder_heads:
                dec_head.init_counter()
        return
