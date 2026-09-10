visual_concepts:
  cultivated_field:
    tnls_sources: [旱地, 水浇地]
    tnls_dl3: [9, 10 ,11]
        prompts:
      - cultivated agricultural field viewed from overhead

  crop_rows:
    tnls_sources: [旱地, 水浇地]
    tnls_dl3: [9, 10 ,11]
        prompts:
      - agricultural field with visible crop rows viewed from overhead

  plowed_or_fallow_field:
    tnls_sources: [旱地]
    tnls_dl3: [11]
        prompts:
      - plowed or fallow agricultural field with exposed soil

  regular_orchard_rows:
    tnls_sources: [果园, 茶园, 其他园地]
    tnls_dl3: [12, 13, 14, 15]
        prompts:
      - orchard with regularly arranged fruit trees viewed from overhead

  dense_tree_canopy:
    tnls_sources: [乔木林地, 其他林地]
    tnls_dl3: [16, 17, 18, 19]
    prompts:
      - dense continuous tree canopy viewed from overhead

  sparse_woody_vegetation:
    tnls_sources: [其他林地, 乔木林地]
    tnls_dl3: [19, 16, 17, 18]
        prompts:
      - sparse discontinuous woody vegetation viewed from overhead

  irregular_tree_crowns:
    tnls_sources: [乔木林地]
    tnls_dl3: [16]
        prompts:
      - irregular natural tree crown pattern viewed from overhead

  herbaceous_cover:
    tnls_sources: [其他草地]
    tnls_dl3: [20, 21, 22]
        prompts:
      - continuous herbaceous vegetation viewed from overhead

  agricultural_facility:
    tnls_sources: [设施农用地]
    tnls_dl3: [54]
        prompts:
      - greenhouse or agricultural facility viewed from overhead

  rural_houses:
    tnls_sources: [农村宅基地]
    tnls_dl3: [28]
        prompts:
      - rural houses with courtyards viewed from overhead

  dense_residential:
    tnls_sources: [城镇住宅用地]
    tnls_dl3: [27]
    prompts:
      - dense residential building blocks viewed from overhead

  large_roof_complex:
    tnls_sources: [工业用地, 物流仓储用地, 商业服务业设施用地]
    tnls_dl3: [25, 29, 30, 31, 32, 33, 34, 23, 24]
        prompts:
      - large-roof building complex viewed from overhead

  narrow_rural_road:
    tnls_sources: [农村道路]
    tnls_dl3: [40]
        prompts:
      - narrow rural road viewed from overhead

  wide_paved_road:
    tnls_sources: [公路用地, 城镇村道路用地]
    tnls_dl3: [36, 37, 38, 39 , 41, 42, 43]
    prompts:
      - wide paved road viewed from overhead

  railway_corridor:
    tnls_sources: [铁路用地]
    tnls_dl3: [35]
    prompts:
      - railway corridor with parallel tracks viewed from overhead

  pond_or_reservoir:
    tnls_sources: [坑塘水面, 水库水面, 养殖坑塘]
    tnls_dl3: [45, 46, 47, 48, 49, 51, 52]
    prompts:
      - pond or reservoir viewed from overhead

  river_or_canal:
    tnls_sources: [河流水面, 沟渠]
    tnls_dl3: [44, 50]
    prompts:
      - river or canal viewed from overhead

  bare_or_excavated_surface:
    tnls_sources: [裸土地, 采矿用地, 裸岩石砾地]
    tnls_dl3: [53, 55, 56, 57, 58, 59, 26]
    prompts:
      - bare or excavated ground surface viewed from overhead




``` python
def dll3Normalize(tensor):
    rasterValueMap = {
        # 水田
        "101": "9",
        # 水浇地
        "102": "10",
        #旱地
        "103": "11",
        #果园
        "201": "12",
        # 茶园
        "202": "13",
        # 橡胶园
        "203": "14",
        # 其他园地
        "204": "15",
        # 乔木林地
        "301": "16",
        # 竹林地
        "302": "17",
        # 灌木林地
        "305": "18",
        # 其他林地
        "307": "19",
        # 天然牧草地
        "401": "20",
        # 人工牧草地
        "403": "21",
        # 其他草地
        "404": "22",
        # 商业服务业设施用地
        "501": "23",
        # 物流仓储用地
        "508": "24",
        # 工业用地
        "601": "25",
        # 采矿用地
        "602": "26",
        # 城镇住宅用地
        "701": "27",
        # 农村宅基地
        "702": "28",
        # 机关团体新闻出版社用地
        "801": "29",
        # 科教文卫用地
        "802": "30",
        # 公共设施用地
        "809": "31",
        # 公园与绿地
        "810": "32",
        # 公园与绿地
        "8100": "33",             
        # 特殊
        "9": "34",
        # 铁路用地
        "1001": "35",
        # 轨道交通用地
        "1002": "36",
        # 铁路用地
        "1003": "37",
        # 城镇村道路用地
        "1004": "38",
        # 交通服务场站用地
        "1005": "39",
        # 农村道路
        "1006": "40",
        # 机场用地
        "1007": "41",
        # 港口码头用地
        "1008": "42",
        # 管道运输用地
        "1009": "43",
        # 河流水面
        "1101": "44",
        # 湖泊水面
        "1102": "45",
        # 水库水面
        "1103": "46",
        #坑塘水面
        "1104": "47",        
        "1105": "48",
        "1106": "49",
        # 沟渠
        "1107": "50",        
        # 水工建筑用地
        "1109": "51",
        # 冰川及永久积雪
        "1110": "52",   
        # 空闲地             
        "1201": "53",
        # 设施农用地
        "1202": "54",
        # 田坎
        "1203": "55",
        # 盐碱地
        "1204": "56",
        # 沙地
        "1205": "57",
        # 裸土地
        "1206": "58",
        # 裸岩石砾地
        "1207": "59",                                
    }

    for key in rasterValueMap.keys():
        tensor[tensor == int(key)] = int(rasterValueMap[key])
    return tensor
```