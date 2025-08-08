import torch
import argparse
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Model import Model, vgae_encoder, vgae_decoder, vgae, DenoisingNet, adversary
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict
import os
from copy import deepcopy
import random
import sys


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        recallMax_top10 = 0
        bestEpoch_top10 = 0
        recallMax_top20 = 0
        bestEpoch_top20  =0
        recallMax_top30 = 0
        bestEpoch_top30 = 0

        stloc = 0
        log('Model Initialized')
        for ep in range(stloc, args.epoch):
            temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch(temperature)
            log(self.makePrint('Train', ep+1, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                if (reses['Recall_top30'] > recallMax_top30):
                    recallMax_top30 = reses['Recall_top30']
                    ndcgMax_top30 = reses['NDCG_top30']
                    dprMax_top30 = reses['DPR_top30']
                    dpnMax_top30 = reses['DPN_top30']
                    bestEpoch_top30 = ep
                if (reses['Recall_top10'] > recallMax_top10):
                    recallMax_top10 = reses['Recall_top10']
                    ndcgMax_top10 = reses['NDCG_top10']
                    dprMax_top10 = reses['DPR_top10']
                    dpnMax_top10 = reses['DPN_top10']
                    bestEpoch_top10 = ep
                if (reses['Recall_top20'] > recallMax_top20):
                    recallMax_top20 = reses['Recall_top20']
                    ndcgMax_top20 = reses['NDCG_top20']
                    dprMax_top20 = reses['DPR_top20']
                    dpnMax_top20 = reses['DPN_top20']
                    bestEpoch_top20 = ep
                log(self.makePrint('Test', ep+1, reses, tstFlag))
                file_path = os.path.join(args.output_dir, 'performance.txt')
                f = open(file_path, "a")
                print(self.makePrint('Test', ep+1, reses, tstFlag),file=f)
            print()

        print('Top10_Best_epoch : ', bestEpoch_top10+1, ' , Recall_top10 : ', round(recallMax_top10,4), ' , NDCG_top10 : ', round(ndcgMax_top10,4), ' DPR_top10: ', round(dprMax_top10,4),' DPN_top10: ', round(dpnMax_top10,4))
        print('Top10_Best_epoch : ', bestEpoch_top10+1, ' , Recall_top10 : ', round(recallMax_top10,4), ' , NDCG_top10 : ', round(ndcgMax_top10,4), ' DPR_top10: ', round(dprMax_top10,4),' DPN_top10: ',round(dpnMax_top10,4), file=f)
        print('Top20_Best_epoch : ', bestEpoch_top20 + 1, ' , Recall_top20 : ', round(recallMax_top20,4), ' , NDCG_top20 : ',round(ndcgMax_top20,4), ' DPR_top20: ', round(dprMax_top20,4), ' DPN_top20: ', round(dpnMax_top20,4))
        print('Top20_Best_epoch : ', bestEpoch_top20 + 1, ' , Recall_top20 : ', round(recallMax_top20,4), ' , NDCG_top20 : ',round(ndcgMax_top20,4), ' DPR_top20: ', round(dprMax_top20,4), ' DPN_top20: ', round(dpnMax_top20,4), file=f)
        print('Top30_Best_epoch : ', bestEpoch_top30+1, ' , Recall_top30 : ', round(recallMax_top30,4), ' , NDCG_top30 : ', round(ndcgMax_top30,4), ' DPR_top30: ', round(dprMax_top30,4),' DPN_top30: ', round(dpnMax_top30,4))
        print('Top30_Best_epoch : ', bestEpoch_top30+1, ' , Recall_top30 : ', round(recallMax_top30,4), ' , NDCG_top30 : ', round(ndcgMax_top30,4), ' DPR_top30: ', round(dprMax_top30,4),' DPN_top30: ', round(dpnMax_top30,4), file=f)
        f.close()

    def prepareModel(self):
        self.model = Model().cuda()

        encoder = vgae_encoder().cuda()
        decoder = vgae_decoder().cuda()
        self.generator_1 = vgae(encoder, decoder).cuda()
        self.adversary = adversary(args.latdim,1).cuda()
        self.criterion_sens = torch.nn.BCEWithLogitsLoss()
        self.generator_2 = DenoisingNet(self.model.getGCN(), self.model.getEmbeds()).cuda()
        self.generator_2.set_fea_adj(args.user+args.item, deepcopy(self.handler.torchBiAdj).cuda())

        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.opt_gen_1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr, weight_decay=0)
        self.opt_gen_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator_2.parameters()), lr=args.lr, weight_decay=0, eps=args.eps)

        self.opt_adv = torch.optim.Adam(self.adversary.parameters(), lr=1e-4, weight_decay=1e-5)



    def trainEpoch(self, temperature):
        senLabel = self.handler.senLabel
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        generate_loss_1, generate_loss_2, bpr_loss, im_loss, ib_loss, reg_loss, adv_loss = 0, 0, 0, 0, 0, 0, 0
        steps = trnLoader.dataset.__len__() // args.batch


        for i, tem in enumerate(trnLoader):
            if (i == 0):
                warmup = 10
            else:
                warmup = 0
            for _ in range(warmup):
                data1 = self.generator_generate(self.generator_1)
                ancs, poss, negs = tem
                ancs = ancs.long().cuda()
                poss = poss.long().cuda()
                negs = negs.long().cuda()

                out1 = self.model.forward_graphcl(data1)  # VAE embedding
                out2 = self.model.forward_graphcl_(self.generator_2)  # denoise embedding

                ssl_loss = self.model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg
                bpr_loss_1 = self.generator_1(deepcopy(self.handler.torchBiAdj).cuda(), ancs, poss, negs)
                bpr_loss_2 = self.generator_2(ancs, poss, negs, temperature)
                loss = ssl_loss + bpr_loss_1 + bpr_loss_2

                self.opt_gen_1.zero_grad()
                self.opt_gen_2.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward(retain_graph=True)
                self.opt_gen_1.step()
                self.opt_gen_2.step()

                # print(
                #     'ssl loss: {:.4f}'.format(ssl_loss.item()),
                #     'bpr loss 1: {:.4f}'.format(bpr_loss_1.item()),
                #     'bpr loss 2: {:.4f}'.format(bpr_loss_2.item()), )


            data = deepcopy(self.handler.torchBiAdj).cuda()
            data1 = self.generator_generate(self.generator_1)

            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            out1 = self.model.forward_graphcl(data1)  # VAE embedding
            out2 = self.model.forward_graphcl_(self.generator_2)  # denoise embedding

            #### update adversary model
            if (i == 0):
                sens_epoches = 10
            else:
                sens_epoches = 1
            for _ in range(sens_epoches):
                user_bedding = self.model.combine_dual_views(out1.detach(), out2.detach(), ancs, poss)  # batch*hidden
                batch_sen_label = torch.tensor([senLabel[i] for i in ancs.tolist()]).cuda()  # batch
                s_pred = self.adversary(user_bedding)
                loss = self.criterion_sens(s_pred, batch_sen_label.unsqueeze(1).float()).mean() * args.adv_reg
                self.opt_adv.zero_grad()
                loss.backward()
                self.opt_adv.step()
            user_bedding1 = self.model.combine_dual_views(out1, out2, ancs, poss)  # batch*hidden
            s_pred = self.adversary(user_bedding1)
            tem_adv_loss = self.criterion_sens(s_pred, batch_sen_label.unsqueeze(1).float()).mean()* args.adv_reg
            #adv_loss += tem_adv_loss
            #print(tem_adv_loss)

            #### update encoder ####

            # info bottleneck
            _out1 = self.model.forward_graphcl(data1)
            _out2 = self.model.forward_graphcl_(self.generator_2)

            loss_ib = self.model.loss_graphcl(_out1, out1.detach(), ancs, poss) + self.model.loss_graphcl(_out2,
                                                                                                          out2.detach(),
                                                                                                          ancs, poss)
            ib_loss = loss_ib.mean() * args.ib_reg
            # ib_loss += float(loss)
            # loss.backward()
            # self.opt.step()
            # self.opt.zero_grad()

            # BPR
            usrEmbeds, itmEmbeds = self.model.forward_gcn(data)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]
            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
            regLoss = calcRegLoss(self.model) * args.reg
            bpr_loss = bprLoss + regLoss
            #bpr_loss += float(bprLoss)
            #reg_loss += float(regLoss)
            #loss.backward()
            ssl_loss = self.model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg
            bpr_loss_1 = self.generator_1(deepcopy(self.handler.torchBiAdj).cuda(), ancs, poss, negs)
            bpr_loss_2 = self.generator_2(ancs, poss, negs, temperature)

            loss = bpr_loss_1 +bpr_loss_2+ssl_loss+ib_loss + bpr_loss - tem_adv_loss
            #print(tem_adv_loss)
            #generate_loss_1 += float(loss_1)
            #generate_loss_2 += float(loss_2)
            #loss.backward()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            log('Step %d/%d: gen 1 : %.3f ; gen 2 : %.3f ; bpr : %.3f ; ssl : %.3f ; ib : %.3f ; reg : %.3f ; adv : %.3f  ' % (
                i,
                steps,
                bpr_loss_1.item(),  #
                bpr_loss_2.item(),  #
                bprLoss.item(),  #
                ssl_loss.item(),  #
                ib_loss.item(),  #
                regLoss.item(),  #
                tem_adv_loss.item(),  #
            ), save=False, oneline=True)
            # print('Epoch: {:04d}'.format(epoch_counter + 1),
            #       'sens loss: {:.4f}'.format(senloss.item()),
            #       'contrastive loss: {:.4f}'.format(contrastive_loss.item()),
            #       'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
            #       'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
            #       )

        ret = dict()
        ret['Gen_1 Loss'] = 0
        ret['Gen_2 Loss'] = 0
        ret['BPR Loss'] = 0
        ret['IM Loss'] = 0
        ret['IB Loss'] = 0
        ret['Reg Loss'] = 0
        ret['Adv Loss'] = 0

        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        senLabel = self.handler.senLabel
        epRecall_k30, epNdcg_k30 = [0] * 2
        epRecall_k10, epNdcg_k10 = [0] * 2
        epRecall_k20, epNdcg_k20 = [0] * 2
        epg0_ndcg_k30, epg1_ndcg_k30, epg0_recall_k30, epg1_recall_k30, epg0_person_k30, epg1_person_k30 = [0] * 6
        epg0_ndcg_k10 , epg1_ndcg_k10 ,epg0_recall_k10, epg1_recall_k10,epg0_person_k10 ,epg1_person_k10 = [0]*6
        epg0_ndcg_k20 , epg1_ndcg_k20 ,epg0_recall_k20, epg1_recall_k20,epg0_person_k20 ,epg1_person_k20 = [0]*6
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            usrEmbeds, itmEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8


            # topk = 10 #
            _, topLocs = torch.topk(allPreds, 10)
            recall, ndcg,g0_ndcg,g1_ndcg, g0_recall,g1_recall,g0_person,g1_person = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, senLabel,10)
            epRecall_k10 += recall
            epNdcg_k10 += ndcg
            epg0_ndcg_k10+= g0_ndcg
            epg1_ndcg_k10 += g1_ndcg
            epg0_recall_k10+= g0_recall
            epg1_recall_k10 += g1_recall
            epg0_person_k10 += g0_person
            epg1_person_k10 += g1_person
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f ' % (i, steps, recall, ndcg), save=False, oneline=True)
            # topk = 20 #
            _, topLocs = torch.topk(allPreds, 20)
            recall, ndcg,g0_ndcg,g1_ndcg, g0_recall,g1_recall,g0_person,g1_person = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, senLabel,20)
            epRecall_k20 += recall
            epNdcg_k20 += ndcg
            epg0_ndcg_k20+= g0_ndcg
            epg1_ndcg_k20 += g1_ndcg
            epg0_recall_k20+= g0_recall
            epg1_recall_k20 += g1_recall
            epg0_person_k20 += g0_person
            epg1_person_k20 += g1_person
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f  ' % (i, steps, recall, ndcg), save=False, oneline=True)

            # topk = 30 #
            _, topLocs = torch.topk(allPreds, 30)
            recall, ndcg,g0_ndcg,g1_ndcg, g0_recall,g1_recall,g0_person,g1_person = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, senLabel,30)
            epRecall_k30 += recall
            epNdcg_k30 += ndcg
            epg0_ndcg_k30+= g0_ndcg
            epg1_ndcg_k30 += g1_ndcg
            epg0_recall_k30+= g0_recall
            epg1_recall_k30 += g1_recall
            epg0_person_k30 += g0_person
            epg1_person_k30 += g1_person
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f ' % (i, steps, recall, ndcg), save=False, oneline=True)
        ret = dict()

        ret['Recall_top10'] = epRecall_k10 / num
        ret['NDCG_top10'] = epNdcg_k10 / num
        ret['DPR_top10'] = abs(epg1_recall_k10/epg1_person_k10-epg0_recall_k10/epg0_person_k10)
        ret['DPN_top10'] = abs(epg1_ndcg_k10/epg1_person_k10-epg0_ndcg_k10/epg0_person_k10)

        ret['Recall_top20'] = epRecall_k20 / num
        ret['NDCG_top20'] = epNdcg_k20 / num
        ret['DPR_top20'] = abs(epg1_recall_k20/epg1_person_k20-epg0_recall_k20/epg0_person_k20)
        ret['DPN_top20'] = abs(epg1_ndcg_k20/epg1_person_k20-epg0_ndcg_k20/epg0_person_k20)

        ret['Recall_top30'] = epRecall_k30 / num
        ret['NDCG_top30'] = epNdcg_k30 / num
        ret['DPR_top30'] = abs(epg1_recall_k30/epg1_person_k30-epg0_recall_k30/epg0_person_k30)
        ret['DPN_top30'] = abs(epg1_ndcg_k30/epg1_person_k30-epg0_ndcg_k30/epg0_person_k30)
        return ret

    def calcRes(self, topLocs, tstLocs, batIds,senlabel,topk):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        g0_ndcg = g1_ndcg = 0
        g0_recall = g1_recall = 0
        g0_person = g1_person = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            if senlabel[batIds[i]] == 0:
                g0_person+=1
                g0_ndcg+=ndcg
                g0_recall+=recall
            else:
                g1_person += 1
                g1_ndcg += ndcg
                g1_recall+=recall
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg, g0_ndcg,g1_ndcg, g0_recall,g1_recall,g0_person,g1_person

    def generator_generate(self, generator):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = deepcopy(self.handler.torchBiAdj)
        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(self.handler.torchBiAdj, idxs, adj)

        return view

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
def  seed(num):
    torch.backends.cudnn.enabled = False
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    logger.saveDefault = True
    seed_it(args.seed)
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    log('output_dir: {}'.format(args.output_dir))
    perf_path = os.path.join(args.output_dir, 'performance.txt')
    parameter_path = os.path.join(args.output_dir, 'parameter.txt')
    f = open(parameter_path, "w")
    for arg in sorted(vars(args)):
            print("{}: {}".format(arg, getattr(args, arg)))
            print("{}: {}".format(arg, getattr(args, arg)), file=f)
    f.close()
    f1 = open(perf_path, "w")
    print("", file=f1)
    f1.close()
    coach = Coach(handler)
    coach.run()