# -*- coding: utf-8 -*-
from db.entity import PredInfo
import pandas as pd
import config as cfg
from db.matchproc import query as mquery
from db.matchproc import bdquery
from db.matchproc import get_engmap, get_match_today, get_match_lastday, get_match_inrange
from db import betproc as bp
from datetime import datetime
import log
import logging
import os
import json
log_file_name = "pred_proc"
log.init_cfg(log_file_name)
logger = logging.getLogger("normal_log")


def _getdf(): 
    '''
    读取预测列表csv
    返回 pandas.dataframe 对象
    '''
    return pd.read_csv(cfg.PRED_RESULT, header=0, index_col=0)


# def get_fet(match_id):
#     # TODO not in this file
#     path = cfg.PROJECT + '/modelfet/%s.json' % str(match_id)
#     fet_dict = None
#     if os.path.isfile(path):
#         with open(path) as f:
#             fet_dict = json.load(f)
#     return fet_dict


def check_pred(match_id, sec):
    '''
    # 查询 N 秒内有记录的识别结果
    '''
    df = _getdf()
    if not (int(match_id) in df['match_id_500'].values):  # 没有任何记录
        print("no record")
        return False
    try:
        ds = df[df['match_id_500'] == int(match_id)]
        match_dict = mquery(match_id)
    except KeyError:
        logger.error("no key %s" % str(match_id))
        return False
    else:

        ds['date_str'] = pd.to_datetime(ds['date_str'])  # 预测的时间列表
        end_time = pd.to_datetime(match_dict['date_str'])  # 比赛结束时间
        start = end_time - pd.Timedelta(seconds=sec)  # 比赛结束时间 - offset
        if pd.Timestamp.now() < start:  # 时间未到
            logger.info("no this time.")
            return True

        end_time_delay = end_time + pd.Timedelta(seconds=600)
        condition1 = ds['date_str'] > start
        condition2 = ds['date_str'] < end_time_delay
        focus_ds = ds[condition1 & condition2]
        logger.info("%s has %d pred logs" % (str(match_id), focus_ds.shape[0]))
        return focus_ds.shape[0] > 0


def _bifen2result(bifen):
    '''
    比分结果转换是否赢 1:赢 0:没有赢 -1 错误结果或未开赛
    '''
    try:
        if bifen == 'VS' or len(bifen) < 3:
            return -1
        msg = bifen.split(':')
    except Exception as e:
        print("fuck bifen", bifen)
        return -1
    return int(int(msg[0]) > int(msg[1]))


def get_diff(d1, d2):
    '''
    返回d1 -d2 按照文字表述
    '''
    t1 = pd.to_datetime(d1)
    t2 = pd.to_datetime(d2)
    tdiff = t1 - t2
    if tdiff.days < 0:
        return "比赛前0秒预测结果"
    diff = tdiff.seconds
    ours, remainder = divmod(diff, 3600)
    minutes, seconds = divmod(diff, 60)
    if ours >= 1:
        return "比赛前{}小时预测结果".format(ours)
    elif minutes >= 1:
        return "比赛前{}分钟预测结果".format(minutes)
    else:
        return "比赛前{}秒预测结果".format(seconds)


def get_preded():
    '''
    获取当日预测
    '''
    df = _getdf()
    mids = list(set(df['match_id_500'].to_list()))
    preds = []
    for mid in mids:
        temp = _query(mid, 6)
        pred_dict = None
        if temp.shape[0] > 0:
            pred_dict = temp.iloc[0, :].to_dict()
        if pred_dict is None or pred_dict["pred"] != 1:
            continue
        match = mquery(mid)
        gamebifen = match['game_bifen']
        result = _bifen2result(gamebifen)
        mkls_1055, t1 = bp.local_bets(mid, 1055)
        if mkls_1055 is None or len(mkls_1055) == 0:
            continue

        pred_dict['result'] = result
        if pred_dict['pred'] == 1:
            if mkls_1055[-1].price <= 0.33 or mkls_1055[-1].price >= 0.67:
                continue
            pred_dict['odds'] = mkls_1055[-1].odds
        else:
            odds_u = 1 / (1 - mkls_1055[-1].price + 0.03)
            price_u = 1/odds_u
            if price_u <= 0.33 or price_u >= 0.67:
                continue

            pred_dict['odds'] = round(odds_u, 2)
        preds.append(pred_dict)
    return preds


def _query(match_id_500, model_id):
    '''
    根据比赛ID，模型ID 返回所有预测结果
    '''
    df = _getdf()
    df['date_str'] = pd.to_datetime(df['date_str'])
    mid = int(match_id_500)
    if not (mid in df['match_id_500'].values):
        return None
    ds_pred = df.loc[(df['match_id_500'] == mid) &
                     (df['model_id'] == model_id)]
    ds_pred = ds_pred.sort_values(by='date_str', ascending=False)
    return ds_pred


def query_pred(model_id):
    df = _getdf()
    df['date_str'] = pd.to_datetime(df['date_str'])
    ds_pred = df.loc[df['model_id'] == model_id]
    print(ds_pred.shape)
    ds_pred = ds_pred.sort_values(by='date_str', ascending=False)
    mids = ds_pred['match_id_500'].tolist()
    mids = list(set(mids))
    results = []
    for mid in mids:
        items = ds_pred[ds_pred['match_id_500'] == mid]
        results.append(items.iloc[0, :].to_dict())
    return results


def insert_pred(pinfo: PredInfo):
    '''
    结构化添加一个预测结果
    '''
    df = _getdf()
    now = datetime.now()
    pinfo.date_str = now.strftime('%Y-%m-%d %H:%M:%S')
    item = pinfo.__dict__
    item.pop('pid')
    df = df.append(item, ignore_index=True)
    df.to_csv(cfg.PRED_RESULT, index_label='pid')


def last10_pred():
    df = _getdf()
    ds = df[df['model_id'] == 6]
    ds = ds.sort_values(by="date_str", ascending=False)
    ds = ds.drop_duplicates(subset=['match_id_500'], keep='first')
    ds = ds[ds['pred'] != -1]
    mids = ds['match_id_500'].to_list()
    displayInfo = []
    for mid in mids:
        predinfo = ds[ds['match_id_500'] == mid].to_dict()
        pred = None
        for k in predinfo['pred']:
            pred = predinfo['pred'][k]
        baseinfo = get_baseInfo(mid, pred)
        if not baseinfo:
            continue
        if baseinfo["result"] == -1:
            print("result -1")
            continue
        if pred is None or pred == -1:
            continue

        if pred == 1:
            baseinfo['预测'] = "主胜"
        else:
            baseinfo['预测'] = "客队不败"

        baseinfo.pop("result")
        displayInfo.append(baseinfo)
    dfdis = pd.DataFrame.from_dict(displayInfo)
    return dfdis


def get_baseInfo(mid, pred=1):
    base_info = {}
    match = mquery(mid)
    if match is None:
        return None
    base_info['联赛'] = match['league_name']
    hometeam = match['home_team_name']
    awayteam = match['away_team_name']
    base_info['比赛时间'] = match['date_str']
    base_info['主队'] = hometeam
    base_info['客队'] = awayteam
    gamebifen = match['game_bifen']
    base_info['比分'] = gamebifen
    result = _bifen2result(gamebifen)
    base_info['result'] = result
    mkls_1055, t1 = bp.local_bets(mid, 1055)
    if mkls_1055 is None or len(mkls_1055) == 0:
        base_info['欧盘'] = 'undefine'
    else:
        if pred:
            cur_price = 1/mkls_1055[-1].odds
        else:
            cur_price = 1 - mkls_1055[-1].price + 0.03
        if cur_price < 0.33 or cur_price > 0.625:
            return None
        base_info['欧盘'] = "%.2f" % (1/cur_price)
    return base_info


def get_predInfo(fliter_fet=False, last_day=False, d1=None, d2=None):
    engmap = get_engmap()
    if d1 and d2:
        matches = get_match_inrange(d1, d2)
    else:
        if last_day:
            matches = get_match_lastday()
        else:
            matches = get_match_today()
    results = []
    for item in matches:
        match_id_500 = item['match_id_500']
        ds_pred = _query(match_id_500, 6)
        if ds_pred is None:
            continue
        if ds_pred.shape[0] == 0:
            continue
        result = {}
        pred2 = ds_pred.iloc[0, :].to_dict()
        pred2['pred'] = int(pred2['pred'])
        pred_hist = ds_pred['pred'].tolist()
        pred_time2 = pred2['date_str'].strftime('%Y-%m-%d %H:%M')
        pred2['date_str'] = pred_time2
        if last_day == False and d1 == None and d2 == None:
            pred2.pop('match_id_500')
        result['pred2'] = pred2
        tmps = []
        for pred_row in pred_hist:
            tmps.append(int(pred_row))
        result['pred2_hist'] = tmps
        itemcp = item['data'].copy()
        itemcp.pop('spdex_id')
        result['match_info'] = itemcp
        if item['data']['away_team_name'] in engmap:
            result['eaway_team_name'] = engmap[item['data']['away_team_name']]
        if item['data']['home_team_name'] in engmap:
            result['ehome_team_name'] = engmap[item['data']['home_team_name']]
        if fliter_fet:
            fet_path = cfg.PROJECT + '/modelfet/' + f"{match_id_500}" + ".json"
            if os.path.isfile(fet_path):
                with open(fet_path) as f:
                    fet_data = json.load(f)
                result['fet'] = fet_data
        results.append(result)
    return results


def get_model_preds(model_id=6):
    matches = get_match_today()  # 获取比赛列表
    results = []
    for item in matches:
        match_id_500 = item['match_id_500']
        ds_pred = _query(match_id_500, model_id)
        if ds_pred is None:
            continue
        if ds_pred.shape[0] == 0:
            continue
        pred2 = ds_pred.iloc[0, :].to_dict()
        pred_time2 = pred2['date_str'].strftime('%Y-%m-%d %H:%M')
        pred2['date_str'] = pred_time2
        results.append(pred2)
    return results


def get_home_win():
    p_rank = {}
    with open(cfg.PROJECT + '/base/pleague.txt') as f:
        for row in f:
            msg = row[:-1].split(',')
            p_rank[msg[0]] = msg[-1]
    blacklist = []
    with open(cfg.PROJECT + '/base/blacklist.txt') as f:
        for row in f:
            blacklist.append(row[:-1])
    results = []
    pred_data = get_model_preds(model_id=6)
    for predinfo in pred_data:
        match_id_500 = predinfo['match_id_500']
        pred = predinfo['pred']
        if pred != 1:
            continue
        fet_path = cfg.PROJECT + '/modelfet/' + f"{match_id_500}" + ".json"
        if os.path.isfile(fet_path):
            with open(fet_path) as f:
                fet_data = json.load(f)
            price = fet_data['endWinPrice']
            if price < 0.3 or price > 0.675:
                continue
        else:
            continue
        minfo = mquery(match_id_500)
        if minfo is None:
            continue
        ln = minfo['league_name']
        hn = minfo['home_team_name']
        an = minfo['away_team_name']
        gt = minfo['date_str']
        game_title = f"{hn}vs{an}"
        t1 = pd.to_datetime(gt)
        t2 = pd.to_datetime(predinfo['date_str'])
        time_msg = get_diff(t1, t2)
        result = {'预测': '主胜'}
        result['对阵双方'] = game_title
        result['比赛时间'] = gt
        result['联赛'] = ln
        result['预测时间'] = time_msg
        if (hn in blacklist) or (an in blacklist):
            result['信心'] = 1
        else:
            if ln in p_rank:
                result['信心'] = p_rank[ln]
            else:
                result['信心'] = 1
        results.append(result)
    if len(results) == 0:
        return None
    ds = pd.DataFrame.from_dict(results)
    ds = ds[[u'联赛', u'对阵双方', u'比赛时间', u'预测', u'预测时间', u'信心']]
    return ds


def get_bei_dian():
    results = []
    pred_map = {0: "10", 1: "30", 2: "31"}
    pred_data = get_model_preds(model_id=6)  # 主胜
    hw_map = {}
    for predinfo in pred_data:
        match_id_500 = predinfo['match_id_500']
        hw_map[match_id_500] = predinfo
    pred_beidan = get_model_preds(model_id=7)  # 北单
    for predinfo in pred_beidan:
        match_id_500 = predinfo['match_id_500']
        bitem = bdquery(match_id_500)
        if bitem is None:
            continue
        minfo = mquery(match_id_500)
        if minfo is None:
            continue
        # print('pinfo',predinfo)
        pred = int(predinfo['pred'])  # 北单算法
        if pred == -1:
            continue

        code = bitem['bd_id']
        rq = int(bitem['rq'])
        # print(f"beidan: {pred}")
        pred_str = pred_map[pred]
        if match_id_500 in hw_map:
            # print(f"home win map {hw_map[match_id_500]}")
            hw_pred = hw_map[match_id_500]['pred']
            if hw_pred == 1:
                if rq == -1:
                    pred_str = "31"
                elif rq == 0:
                    pred_str = "3"
                else:
                    continue
            else:
                # print(f"让球 {rq}")
                if rq != 0:
                    continue
        else:
            if rq != 0:
                continue
        ln = minfo['league_name']
        hn = minfo['home_team_name']
        an = minfo['away_team_name']
        gt = minfo['date_str']
        game_title = f"{hn}vs{an}"
        t1 = pd.to_datetime(gt)
        t2 = pd.to_datetime(predinfo['date_str'])
        time_msg = get_diff(t1, t2)
        result = {'北单编号': str(code)}
        result['预测'] = pred_str
        result['对阵双方'] = game_title
        result['让球'] = str(rq)
        result['比赛时间'] = gt
        result['联赛'] = ln
        result['预测时间'] = time_msg
        results.append(result)

    if len(results) == 0:
        return None
    ds = pd.DataFrame.from_dict(results)
    ds = ds[[u'北单编号', u'联赛', u'对阵双方', '让球', u'比赛时间', u'预测时间', u'预测']]
    return ds


def get_lota_json():
    results = []
    pred_map = {0: "10", 1: "30", 2: "31"}
    pred_data = get_model_preds(model_id=6)  # 主胜
    hw_map = {}
    for predinfo in pred_data:
        match_id_500 = predinfo['match_id_500']
        hw_map[match_id_500] = predinfo
    pred_beidan = get_model_preds(model_id=7)  # 北单
    for predinfo in pred_beidan:
        match_id_500 = predinfo['match_id_500']
        bitem = bdquery(match_id_500)
        if bitem is None:
            continue
        minfo = mquery(match_id_500)
        if minfo is None:
            continue
        pred = int(predinfo['pred'])  # 北单算法
        if pred == -1:
            continue
        code = bitem['bd_id']
        issue_num = bitem['issue_num']
        rq = int(bitem['rq'])
        pred_str = pred_map[pred]
        if match_id_500 in hw_map:
            hw_pred = hw_map[match_id_500]['pred']
            if hw_pred == 1:  # lota 强制刷新让球0
                if rq == -1:
                    rq = 0
                    pred_str = "3"
                elif rq == 0:
                    pred_str = "3"
                else:
                    continue
            else:
                if rq != 0:
                    continue
        else:
            if rq != 0:
                continue
        ln = minfo['league_name']
        hn = minfo['home_team_name']
        an = minfo['away_team_name']
        gt = minfo['date_str']
        pred_time = pd.to_datetime(
            predinfo['date_str']).strftime('%Y-%m-%d %H:%M:%S')
        result = {'number': code}
        result['issue_number'] = issue_num
        result['pred'] = pred_str
        result['home_team_name'] = hn
        result['away_team_name'] = an
        result['handicap'] = str(rq)
        result['game_time'] = gt
        result['league_name'] = ln
        result['pred_time'] = pred_time我
        results.append(result)
    if len(results) == 0:
        return None
    logger.info(f"LOTA json :{results}")
    return results
