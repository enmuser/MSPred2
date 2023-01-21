"""
Spatio-Temporal Hierarchical Multi-Scale Stochastic Video Prediction model.
This corresponds to our proposed MSPred model
"""

import torch
import torch.nn as nn
import models


class SpatioTempHierarch(models.HierarchModel):
    """ MSPred model """

    def __init__(self, model_params, linear=True, stochastic=False, use_output=True, **kwargs):
        """ MSPred model intializer """ # linear = False
        super().__init__(model_params=model_params, linear=linear,
                         stochastic=stochastic, use_output=use_output, **kwargs)
        assert self.num_hierarch == 3, "Invalid number of hierarchy levels!"
        assert not (self.last_context_residuals and self.linear), "Residual-connections used with linear-LSTMs!"
        assert not (self.ancestral_sampling and self.linear), "Ancestral-sampling used with linear-LSTMs!"
        return

    def predict(self, feats_dict, out_dict, cur_frame, context):
        """
        Predicting next features using current ones. Corresponds to the forward # 使用当前特征预测下一个特征。 对应于通过我们的分层预测器模块的前向传递。
        pass through our hierarchical predictor module.

        Args:
        -----
        feats_dict: dict
            Dict. containing tensor with the input and target features for each level in the hiearchy  # 包含具有层次结构中每个级别的输入和目标特征的张量
        out_dict: dict
            Dict containing lists where all outputs and intermediate values are stored # 包含存储所有输出和中间值的列表的字典
        cur_frame: int
            Index of the current frame in the sequence # 序列中当前帧的索引
        context: int
            Number of context frames to use # 要使用的上下文帧数
        """
        feats = feats_dict["input"] # feats = [16 * 128 * 16 * 16, 16 * 256 * 8 * 8, 16 * 512 * 4 * 4]
        target_feats = feats_dict["target"] # target_feats = [16 * 128 * 16 * 16, 16 * 256 * 8 * 8, 16 * 512 * 4 * 4]

        # sampling latent vectors for each level in the hierarchy using Gaussian LSTMs # 使用高斯 LSTM 对层次结构中的每个级别进行潜在向量采样
        if (self.stochastic): # True
            for h in reversed(range(self.num_hierarch)):# 2,1,0
                post_input = target_feats[h] # post_input =[16 * 512 * 4 * 4, 16 * 256 * 8 * 8, 16 * 128 * 16 * 16]
                prior_input = feats[h] # prior_input =[16 * 512 * 4 * 4, 16 * 256 * 8 * 8, 16 * 128 * 16 * 16]
                # Ancestral sampling: each level latent also conditioned on all upper level samples. 祖先采样：每个潜在级别也以所有上层样本为条件。
                if self.ancestral_sampling and h != self.num_hierarch-1: # ancestral_sampling = True And [2 != 2, 1 != 2, 0 != 2] = [False,True,True]
                    prev_latents = [latent_list[-1] for latent_list in out_dict["latents"][h+1:]] # 找到上一层的latents 倒序组成数组 1找2 0找1
                    # Spatially upsample 2x, 4x, ... and concatenate latent samples from upper levels.  # 当h=1时 prev_latents=[16 * 10 * 4 * 4]
                    for i, z in enumerate(prev_latents):                                               # 当h=0时 prev_latents=[16 * 10 * 8 * 8,16 * 10 * 4 * 4]
                        prev_latents[i] = nn.functional.interpolate(z, scale_factor=2**(i+1)) # 插值上采样 当h=1时 prev_latents=[16 * 10 * 4 * 4] => [16 * 10 * 8 * 8]
                    prev_latents = torch.cat(prev_latents, dim=1)  # 插值上采样  当h=0时 prev_latents=[16 * 10 * 8 * 8,16 * 10 * 4 * 4] => [16 * 10 * 16 * 16,16 * 10 * 16 * 16]
                    post_input = torch.cat([post_input, prev_latents], dim=1) # 当h=1时 cat(16 * 256 * 8 * 8,16 * 10 * 8 * 8) => 16 * 266 * 8 * 8 //// 当h=0时  cat( 16 * 128 * 16 * 16,16 * 20 * 16 * 16) => 16 * 148 * 16 * 16
                    prior_input = torch.cat([prior_input, prev_latents], dim=1) # 当h=1时 cat(16 * 256 * 8 * 8,16 * 10 * 8 * 8) => 16 * 266 * 8 * 8 //// 当h=0时 cat( 16 * 128 * 16 * 16,16 * 20 * 16 * 16) => 16 * 148 * 16 * 16

                # forward through prior and posterion gaussian LSTMs # 通过先验和后验高斯 LSTM 向前
                (latent_post, mu_post, logvar_post), ticked = self.posterior[h](post_input) # (16 * 10 * 4 * 4,16 * 10 * 4 * 4,16 * 10 * 4 * 4),True
                if not ticked: #  (16 * 10 * 8 * 8,16 * 10 * 8 * 8,16 * 10 * 8 * 8)
                    continue  #  (16 * 10 * 16 * 16,16 * 10 * 16 * 16,16 * 10 * 16 * 16)
                (latent_prior, mu_prior, logvar_prior), _ = self.prior[h](prior_input) # (16 * 10 * 4 * 4,16 * 10 * 4 * 4,16 * 10 * 4 * 4),True
                latent = latent_post if (cur_frame < context-1 or self.training) else latent_prior
                out_dict["latents"][h].append(latent)  # (cur_frame < 17 - 1 = 16 or training =True) = True
                if cur_frame >= context-1: # cur_frame >= 17 -1 = 16
                    out_dict["mu_post"][h].append(mu_post)
                    out_dict["logvar_post"][h].append(logvar_post)
                    out_dict["mu_prior"][h].append(mu_prior)
                    out_dict["logvar_prior"][h].append(logvar_prior)

        assert len(feats) >= len(self.predictor)
        # predicting next features for each level in the hierarchy
        pred_outputs = []
        for h, cur_model in enumerate(self.predictor):
            feats_ = feats[h] # feats_ = [16 * 128 * 16 * 16, 16 * 256 * 8 * 8, 16 * 512 * 4 * 4]
            if self.stochastic:
                latent = out_dict["latents"][h][-1] # latent =[16 * 10 * 16 * 16,16 * 10 * 8 * 8,16 * 10 * 4 * 4]
                if self.linear: # False
                    feats_ = torch.cat([feats_.reshape(-1, cur_model.input_size-latent.shape[1]), latent], 1)
                else:# cat(16 * 128 * 16 * 16,16 * 10 * 16 * 16) = 16 * 138 * 16 * 16 //// cat(16 * 256 * 8 * 8,16 * 10 * 8 * 8) = 16 * 266 * 8 * 8
                    feats_ = torch.cat([feats_, latent], 1) # feats_ = [16 * 138 * 16 * 16,16 * 266 * 8 * 8,16 * 522 * 4 * 4]
            pred_feats = cur_model(feats_, hidden_state=cur_model.hidden[0])
            pred_outputs.append(pred_feats) # pred_feats = [16 * 128 * 16 * 16,16 * 256 * 8 * 8,16 * 512 * 4 * 4]
        if self.linear: # False
            pred_outputs = self._reshape_preds(pred_outputs)

        return pred_outputs, out_dict # pred_outputs =[16 * 128 * 16 * 16,16 * 256 * 8 * 8,16 * 512 * 4 * 4]

    def _reshape_preds(self, preds):
        """ Reshaping predicted feature vectors before passing to decoder """
        assert self.linear is True
        assert len(preds) == self.num_hierarch
        img_size = self.model_params["img_size"]
        nf_enc = self.model_params["enc_dec"]["num_filters"]
        for n in range(self.num_hierarch):
            C = nf_enc * 2**(n+1)
            H, W = self.encoder.get_spatial_dims(img_size, n+1)
            preds[n] = torch.reshape(preds[n], (-1, C, H, W))
        return preds

    def _get_input_feats(self, enc_outs, enc_skips):
        return [*enc_skips[1:], enc_outs]

    def _get_residual_feats(self, enc_outs, enc_skips):
        return [*enc_skips, enc_outs] #
                                                          # pred_feats预测的结果 =[16 * 128 * 16 * 16,16 * 256 * 8 * 8,16 * 512 * 4 * 4]
    def _get_decoder_inputs(self, pred_feats, residuals): # residuals输入的特征=[16 * 64 * 32 * 32, 16 * 128 * 16 * 16, 16 * 256 * 8 * 8, 16 * 512 * 4 * 4]
        dec_input_feats = [residuals[0]] # dec_input_feats = [16 * 64 * 32 * 32]
        if not self.last_context_residuals: # False
            dec_input_feats = dec_input_feats + pred_feats
        else:
            for i, feat in enumerate(pred_feats):
                dec_input_feats.append(torch.add(feat, residuals[i+1])) # dec_input_feats = [16 * 64 * 32 * 32, 16 * 128 * 16 * 16, 16 * 256 * 8 * 8, 16 * 512 * 4 * 4]
        return [dec_input_feats[-1], dec_input_feats[:-1]] # return => [16 * 512 * 4 * 4,[16 * 64 * 32 * 32, 16 * 128 * 16 * 16, 16 * 256 * 8 * 8]]

    def _get_decoder_head_inputs(self, pred_feats, dec_skips):
        return dec_skips[-2::-1] # pred_feats = [16 * 128 * 16 * 16,16 * 256 * 8 * 8,16 * 512 * 4 * 4]
                                 # dec_skips =[16 * 256 * 8 * 8,16 * 128 * 16 * 16,16 * 64 * 32 * 32] => return [16 * 128 * 16 * 16,16 * 256 * 8 * 8]
    def _get_predictor(self):
        """ Instanciating the temporal-hierarchy prediction model """ # 实例化时间层次预测模型
        pred_model = models.LSTM if self.linear else models.ConvLSTM # pred_model = models.ConvLSTM
        nf_enc = self.model_params["enc_dec"]["num_filters"] # nf_enc = 64
        img_size = self.model_params["img_size"] # img_size = (64,64)

        predictor = []
        for n in range(self.num_hierarch): # num_hierarch = 3,[0,1,2]
            input_size = output_size = nf_enc * 2**(n+1) # input_size = output_size = [64 * 2,64 * 4,64 * 8] = [128, 256, 512]
            if self.linear:
                h, w = self.encoder.get_spatial_dims(img_size, n+1)
                input_size = input_size * (h * w)
            output_size = input_size
            if self.stochastic:
                input_size += self.model_params["LSTM_Prior"]["latent_dim"] # input_size = [128, 256, 512] + 10 = [138,266,522]
            predictor.append(
                pred_model(
                    input_size=input_size,# input_size = [138,266,522]
                    output_size=output_size, # output_size = [128,256,512]
                    hidden_size=self.hidden_size,# hidden_size = 128
                    num_layers=self.num_lstm_layers[n],# num_lstm_layers = [4,4,4]
                    period=self.periods[n], # periods = [1,4,8]
                    use_output=self.use_output # use_output =True
                )
            )
        predictor = nn.ModuleList(predictor)
        return predictor

    def _get_prior_post(self, model_key):
        """
        Instanciating the modules to estimate the posterior and prior distributions.
        We use different recurrent models for each level of the temporal hierarchy.
        """
        assert model_key in ["LSTM_Prior", "LSTM_Posterior"] # linear = False
        prior_model = models.Gaussian_LSTM if self.linear else models.GaussianConvLSTM # GaussianConvLSTM
        nf_enc = self.model_params["enc_dec"]["num_filters"] # nf_enc = 64
        z_dim = self.model_params["LSTM_Prior"]["latent_dim"] # z_dim = 10
        hid_dim = self.model_params[model_key]["hidden_dim"] # hid_dim = 64
        n_layers = self.model_params[model_key]["num_layers"] # n_layers = 1
        img_size = self.model_params["img_size"] # img_size =(64,64)
        modules = []
        for n in range(self.num_hierarch): # num_hierarch = 3, [0,1,2]
            input_size = nf_enc * 2**(n+1) # input_size= [64 * 2, 64 * 4,64 * 8] = [128,256,512]
            if self.linear: # False
                h, w = self.encoder.get_spatial_dims(img_size, n+1)
                input_size = input_size * (h * w)
            if self.ancestral_sampling: # True
                input_size += (z_dim * (self.num_hierarch-n-1)) # input_size =  [128,256,512] + [10 * 2, 10 * 1, 0] = [148,266,512]
            modules.append(
                prior_model(
                    input_size=input_size,
                    output_size=z_dim,
                    hidden_size=hid_dim,
                    num_layers=n_layers,
                    period=self.periods[n]
                )
            )
        model = nn.ModuleList(modules)
        return model

    def _get_decoder_heads(self):
        """ Instanciating decoder heads for predicting high-level representations """ # 实例化解码器头以预测高级表示
        nf = self.model_params["enc_dec"]["num_filters"] #  nf = 64
        out_channels = self.n_hmap_channels # out_channels = [1,1]
        decoder_heads = nn.ModuleList()
        for h in range(1, self.num_hierarch): # num_hierarch = 3, [1,2]
            decoder_heads.append(
                    models.DeconvHead(
                            in_channels=nf*(2**h), # in_channels = [64 * 2,64 * 4] = [128,256]
                            num_filters=nf, # nf= 64
                            out_channels=out_channels[h-1],# out_channels = [1,1]
                            num_layers=h+1,# num_layers = [2,3]
                            period=self.periods[h] # periods = [4,8]
                        )
                )
        return decoder_heads

    def init_hidden(self, batch_size):
        """ Initializing hidden states for all recurrent models """
        device = self.device_param.device
        img_size = self.model_params["img_size"] # img_size = (64,64)
        for h, m in enumerate(self.predictor):
            _ = m.init_hidden(
                    batch_size=batch_size,
                    device=device,
                    input_size=self.encoder.get_spatial_dims(img_size, h+1) # img_size = (64,64),[1,2,3] => [(16,16),(8,8),(4,4)]
                )
        if self.stochastic: # True
            for h, m in enumerate(self.prior):
                _ = m.init_hidden(
                        batch_size=batch_size,
                        device=device,
                        input_size=self.encoder.get_spatial_dims(img_size, h+1)
                    )
            for h, m in enumerate(self.posterior):
                _ = m.init_hidden(
                        batch_size=batch_size,
                        device=device,
                        input_size=self.encoder.get_spatial_dims(img_size, h+1)
                    )
        return

#
