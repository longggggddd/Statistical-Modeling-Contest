# -*-coding:utf-8-*-
import csv
import requests
from lxml import etree
from concurrent.futures import ThreadPoolExecutor


def get_fangyuan_url(url_out, headers):
    resp_out = requests.get(url_out, headers=headers)
    xp_out_object = etree.HTML(resp_out.text)
    xp_url_lst = xp_out_object.xpath('//*[@id="Lmain_con"]/li/div/div[1]/div[1]/div[1]/a/@href')
    return ['https://www.muniao.com' + i for i in xp_url_lst]


def get_fangdong_url(url_out, headers):
    resp_out = requests.get(url_out, headers=headers)
    xp_out_object = etree.HTML(resp_out.text)
    landlord = xp_out_object.xpath('//*[@id="Lmain_con"]/li/div/div[1]/div[2]/div[1]/div[1]/a/@href')
    return ['https://www.muniao.com' + i for i in landlord]


def get_cont1(url_in, headers):
    resp_in = requests.get(url_in, headers=headers)
    xp_in_object = etree.HTML(resp_in.text)
    name = xp_in_object.xpath('//*[@id="room_mainbox"]/div[1]/h1/span/a/text()')[0]
    # id = xp_in_object.xpath('//*[@id="room_mainbox"]/div[2]/text()')[0]  # //*[@id="room_mainbox"]/div[2]/text()
    describe = xp_in_object.xpath('//*[@id="room_nrbox"]/div[2]/text()')[0]
    rate_of_house_good_comment = xp_in_object.xpath('//*[@id="room_mainbox"]/div[3]/ul[2]/li[1]/p[1]/text()')[0]
    rate_of_house_good_comment = rate_of_house_good_comment.strip()
    predetermined_amount = xp_in_object.xpath('//*[@id="room_mainbox"]/div[3]/ul[2]/li[2]/p[1]/text()')[0]
    predetermined_amount = predetermined_amount.strip()
    property_type = xp_in_object.xpath('//*[@id="fjxx"]/li[1]/div/span/text()')[0].strip()
    hu_xing = xp_in_object.xpath('//*[@id="fjxx"]/li[2]/div/span/text()')[0].strip()
    chu_zu_lei_xing = xp_in_object.xpath('//*[@id="fjxx"]/li[3]/div/span/text()')[0].strip()
    chuang_xing = xp_in_object.xpath('//*[@id="fjxx"]/li[4]/div/span/text()')[0].strip()
    chuang_shu = xp_in_object.xpath('//*[@id="fjxx"]/li[5]/div/span/text()')[0].strip()
    ke_zhu_ren_shu = xp_in_object.xpath('//*[@id="fjxx"]/li[6]/div/span/text()')[0].strip()
    du_wei_shu_liang = xp_in_object.xpath('//*[@id="fjxx"]/li[7]/div/span/text()')[0].strip()
    mian_ji = xp_in_object.xpath('//*[@id="fjxx"]/li[8]/div/span/text()')[0].strip()
    fa_piao = xp_in_object.xpath('//*[@id="fjxx"]/li[9]/div/span/text()')[0].strip()
    address = xp_in_object.xpath('//*[@id="ass"]/text()')[0].strip()
    sheng_fen = address.split('-')[0]
    area = address.split('-')[1]
    pei_tao_she_shi = xp_in_object.xpath('//*[@id="ptss"]/li/span[2]/text()')
    zhou_bian_she_shi = xp_in_object.xpath('//*[@id="room_near"]/div[2]/text()')[0].strip()
    cheng_che_lu_xian = xp_in_object.xpath('//*[@id="room_near"]/div[2]/text()')[1].strip()
    if len(xp_in_object.xpath('//*[@id="room_mainbox2"]/div[11]/div[2]/ul/li/div[2]/div[2]/text()')) > 0:
        fang_yuan_ping_lun_shu = len(
            xp_in_object.xpath('//*[@id="room_mainbox2"]/div[11]/div[2]/ul/li/div[2]/div[2]/text()'))

        # //*[@id="room_mainbox2"]/div[12]/div[2]/ul/li/div[2]/div[2]
        # //*[@id="room_mainbox2"]/div[12]/div[2]/ul/li[1]/div[2]/div[2]

        fang_ke_ping_jia = "+".join(
            xp_in_object.xpath('//*[@id="room_mainbox2"]/div[11]/div[2]/ul/li/div[2]/div[2]/text()'))

    else:
        fang_yuan_ping_lun_shu = len(
            xp_in_object.xpath('//*[@id="room_mainbox2"]/div[12]/div[2]/ul/li/div[2]/div[2]/text()'))

        # //*[@id="room_mainbox2"]/div[12]/div[2]/ul/li/div[2]/div[2]
        # //*[@id="room_mainbox2"]/div[12]/div[2]/ul/li[1]/div[2]/div[2]

        fang_ke_ping_jia = '+'.join(
            xp_in_object.xpath('//*[@id="room_mainbox2"]/div[12]/div[2]/ul/li/div[2]/div[2]/text()'))

    jiege = xp_in_object.xpath('//*[@id="rentform"]/div[2]/div/span[1]/text()')[0].strip()

    try:
        zonghedefen = xp_in_object.xpath('//*[@id="room_mainbox2"]/div[11]/div[2]/div/div[2]/div/text()')
        if len(zonghedefen) > 0:
            zonghedefen = xp_in_object.xpath('//*[@id="room_mainbox2"]/div[11]/div[2]/div/div[2]/div/text()')[0].strip()
        else:
            zonghedefen = xp_in_object.xpath('//*[@id="room_mainbox2"]/div[12]/div[2]/div/div[2]/div/text()')[0].strip()
    except:
        zonghedefen = ''

    return [name, describe, rate_of_house_good_comment, predetermined_amount, property_type, hu_xing,
            chu_zu_lei_xing, chuang_xing,
            chuang_shu, ke_zhu_ren_shu, du_wei_shu_liang, mian_ji, fa_piao, address, sheng_fen, area, pei_tao_she_shi,
            zhou_bian_she_shi,
            cheng_che_lu_xian, fang_yuan_ping_lun_shu, fang_ke_ping_jia, jiege, zonghedefen]


def get_cont2(url_in, headers):
    resp_fangzhu = requests.get(url_in, headers=headers)
    xp_fangzhu = etree.HTML(resp_fangzhu.text)
    fang_zhu_hao_ping_lv = xp_fangzhu.xpath('/html/body/div[5]/div[1]/div/div[2]/div[4]/div[1]/span/text()')[0]
    fang_zhu_hui_fu_lv = xp_fangzhu.xpath('/html/body/div[5]/div[1]/div/div[2]/div[4]/div[2]/span/text()')[0]
    fang_zhu_jie_dan_lv = xp_fangzhu.xpath('/html/body/div[5]/div[1]/div/div[2]/div[4]/div[3]/span/text()')[0]
    qi_ta_fang_yuan_shu = 0
    qi_ta_fang_yuan_ping_jun_fen = []
    for i in range(10000):
        try:
            a = xp_fangzhu.xpath(f'//*[@id="div_1"]/div[{i + 1}]/div[5]/div[3]/text()')[0]
            qi_ta_fang_yuan_ping_jun_fen.append(a)
            qi_ta_fang_yuan_shu += 1
        except:
            break
    qi_ta_fang_yuan_shu = qi_ta_fang_yuan_shu
    qi_ta_fang_yuan_ping_jun_fen = sum([float(i) for i in qi_ta_fang_yuan_ping_jun_fen]) / qi_ta_fang_yuan_shu
    return [fang_zhu_hao_ping_lv, fang_zhu_hui_fu_lv, fang_zhu_jie_dan_lv, qi_ta_fang_yuan_shu,
            round(qi_ta_fang_yuan_ping_jun_fen, 3)]


def main(address='tianjin', page=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"
    }
    feature_name = ['名称', '描述', '房源好评率', '近期预定量', '房源类型', '户型', '出租类型', '床型', '床数', '可住人数', '独卫数量', '面积', '发票',
                    '地址', '省份', '城区',
                    '配套设施',
                    '周边设施', '乘车路线',
                    '房源评论数', '房客评价', '价格', '房源综合得分', '房主好评率', '房主回复率', '房主接单率', '其他房源数', '其他房源平均分']
    with open(f"{address}.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(feature_name)
        for j in range(1, page + 1):
            url_out = f'https://www.muniao.com/{address}/null-0-0-0-0-0-0-0-{j}.html?tn=mn19091015'
            for i in range(len(get_fangyuan_url(url_out, headers))):
                try:
                    fangyuan_url = get_fangyuan_url(url_out, headers)[i]
                    fangdong_url = get_fangdong_url(url_out, headers)[i]
                    get_cont_1 = get_cont1(fangyuan_url, headers)
                    get_cont_2 = get_cont2(fangdong_url, headers)
                    get_cont_1.extend(get_cont_2)
                    wr.writerow(get_cont_1)
                    print(f'第{j}页第{i + 1}个民宿爬取完成...')
                except:
                    print(f'第{j}页第{i + 1}个民宿爬取失败...')
                    continue
            print(f'第{j}页爬取完成！！')


if __name__ == '__main__':
    ADDRESS = 'hengshui'  # 城市拼音
    PAGE = 10  # PAGE 最多是10页 每页30个民宿 所以一个城市最多能爬取300个民宿信息
    main(address=ADDRESS, page=PAGE)
