from flask import Flask, jsonify

app = Flask(__name__)

plant_inf={
    '艾叶':'艾叶，中药名。为菊科植物艾Artemisia argyi Levl.et Vant.的干燥叶。夏季花未开时采摘，除去杂质，晒干。',
    '巴戟天':'巴戟天（学名：Morinda officinalis How）是茜草科、巴戟天属植物。藤本；肉质根不定位肠状缢缩，根肉略紫红色，干后紫蓝色；嫩枝被长短不一粗毛。叶薄或稍厚，纸质，干后棕色，长圆形，卵状长圆形或倒卵状长圆形。花序3-7伞形排列于枝顶；花序梗长5-10毫米，被短柔毛；头状花序具花4-10朵；；花柱外伸，柱头长圆形或花柱内藏，柱头不膨大，2等裂或2不等裂，子房（2-）3（-4）室，每室胚珠1颗，着生于隔膜下部。聚花核果由多花或单花发育而成，熟时红色，扁球形或近球形，直径5-11毫米；核果具分核（2-）3（-4）；分核三棱形，外侧弯拱，被毛状物，内面具种子1，果柄极短；种子熟时黑色，略呈三棱形，无毛。花期5-7月，果熟期10-11月。',
    '白花蛇舌草':'白花蛇舌草（学名：Hedyotis diffusa Willd.）为一年生披散草本，高15-50cm。根细长，分枝，白花。茎略带方形或扁圆柱形，光滑无毛，从基部发出多分枝。花期春季。种子棕黄色，细小，且3个棱角。其成药味苦、淡，性寒。主要功效是清热解毒、消痛散结、利尿除湿。尤善治疗各种类型炎症。在临床实践中，发现白花蛇舌草若配伍得当，可治疗多种疾病。 [1]  别名：蛇舌草、蛇舌癀、蛇针草、蛇总管、二叶葎、白花十字草、尖刀草、甲猛草、龙舌草、蛇脷草、鹤舌草。',
    '白茅根':'白茅根，中药名。为禾本科植物白茅Imperata cylindrica Beauv.var. major（ Nees）C.E.Hubb.的干燥根茎。春、秋二季采挖，洗净，晒干，除去须根和膜质叶鞘，捆成小把。',
    '白芍':'白芍，Cynanchum otophyllum Schneid.是萝藦科鹅绒藤属植物；多年生草质藤本；根圆柱状,灰黑色。生于山地疏林或山坡灌木丛中；海拔1400-2800米分布于大理北部及云南大部分地区；西藏、四川、广西、湖南也有。 [1] 药用根，味辛、苦；有小毒；具温阳祛湿、补体虚、健脾胃等功效。白芍民间用 于治疗风湿冷痛、风湿关节炎、腰肌劳损、体虚神衰、四肢抽搐、慢惊风、犬咬伤等病症。',
    '白头翁':'白头翁（拉丁学名：Pulsatilla chinensis (Bunge) Regel），毛茛科，白头翁属多年生草本植物，长有根状茎，叶片呈卵形，花萼蓝紫色。别名有奈何草、粉乳草、白头草、老姑草等等。分布在中国的吉林、辽宁、河北、山东、河南、山西、陕西、黑龙江等省的山岗、荒坡及田野间。有清热解毒、凉血止痢、燥湿杀虫的功效，具有很高的药用价值。白头翁在园林中可作自然栽植，用于布置花坛、道路两旁，或点缀于林间空地。是理想的地被植物品种。',
    '百部':'百部是百部科百部属多年生攀援性草本植物。地下根为块状根，成束，肉质，为长纺锤形；茎较长；叶为卵形、卵状披针形，顶端渐尖或锐尖，叶有叶柄；花梗紧贴叶片中脉生长，花单生或数朵排列成总状花序； [4]  蒴果卵形，稍扁，表面暗红棕色； [4]  种子椭圆形，紫褐色。花期5-7月，果期7-10月。 [3] 百部分布于中国浙江、江苏、安徽、江西等省，日本曾引入栽培；生长于海拔300-400米的山坡草丛、路旁和林下。 [3]  其主要繁殖方式有有性繁殖中的种子繁殖、无性繁殖中的分根繁殖、无性繁殖中的组织培养三种方法。 [5] 百部最早记载于《名医别录》，被列为中品，在中国药用历史悠久，为中医临床常用中药， [5]  性甘、苦，以粗壮、肥润、坚实、色白者为佳。 [6]  外用可杀虫、止痒、灭虱；内服有润肺、止咳、祛痰之效， [3]  其功效在历代本草著作，例如《本草纲目》《本草经疏》中均有记载。',
    '百合':'百合是百合科百合属植物。地下根茎为鳞茎球状，广展，无明显结节，白色；茎有紫色条纹，无毛；叶散生，上部叶常比中部叶小，倒披针形，叶缘平整，无毛，具有较短的叶柄；花为喇叭形，有香味，多为白色，背面带紫褐色，无斑点，顶端弯而不卷；蒴果矩圆形，有棱，内具多数种子。 花、果期6-9月。 [8]  [14] 百合在中国主要分布于河北、山西、河南等地，生于海拔300-920米的山坡草丛中、疏林下、山沟旁、地边或村旁，适应性较强 [8]  ，喜凉爽、湿润的半阴环境，较耐寒冷，属长日照植物 [2]  ，无性繁殖和有性繁殖均可，生产上主要用鳞片、小鳞茎和珠芽繁殖。 [9]  在中国，百合具有百年好合美好家庭、伟大的爱之含意 [10]  ； 西方的基督教文化中，百合花原本是黄色，被用来象征圣母玛利亚的纯洁后才变成白色',
    '半夏':'半夏（Pinellia ternata (Thunb.) Ten. ex Breitenb.），是天南星科半夏属多年生草本植物。 [1]具块茎，叶基出，有长柄，叶柄基部常有珠芽。肉穗花序具细长附属体；花雌雄同株，无花被；雌花部分与佛焰苞贴生。浆果小，熟时红色。 [4]半夏有两种繁殖方式，有性生殖通过花完成，营养繁殖通过珠芽和块茎完成。 [5-7]因仲夏可采其块茎，故名“半夏”',
    '北沙参':'珊瑚菜（Glehnia littoralis F. Schmidt ex Miq.），渐危种，又名北沙参，其根入药。多年生草本，高5-25厘米。主根细长，圆柱形，分枝，叶基生，白色花瓣，果实呈圆形或卵形',
    '苍术':'苍术（Atractylodes Lancea (Thunb.) DC.）是菊科苍术属多年生草本植物。根状茎平卧或斜升，粗长或通常呈疙瘩状，生多数等粗等长或近等长的不定根。茎直立，单生或少数茎成簇生，全部叶质地硬，两面同色呈绿色，无毛。瘦果倒卵圆状，覆盖稠密的顺向贴伏的白色长直毛',
    '侧柏叶':'侧柏叶，中药名。为柏科植物侧柏PLatycladus orientalis（L.）Franco的干燥枝梢和叶。多在夏、秋二季采收，阴干。',
    '柴胡':'柴胡，中药名。为《中国药典》收录的草药，药用部位为伞形科植物柴胡或狭叶柴胡的干燥根。春、秋二季采挖，除去茎叶及泥沙，干燥。柴胡是常用解表药。别名地熏、山菜、菇草、柴草，性味苦、微寒，归肝、胆经。有和解表里，疏肝升阳之功效。用于感冒发热、寒热往来、疟疾、肝郁气滞、胸肋胀痛、脱肛、子宫脱垂、月经不调。',
    '赤芍':'赤芍，中药名。为毛茛科植物芍药或川赤芍的干燥根。春、秋二季采挖，除去根茎、须根及泥沙，晒干。苦，微寒。归肝经。有清热凉血，活血祛瘀的功效。赤芍是著名野生地道中药材，应用历史悠久，用量较大、用途广泛且需求较为刚性，每年都有相当数量的出口。',
    '穿心莲':'穿心莲（Andrographis paniculata (Burm. f.) Wall. ex Nees in Wallich），爵床科穿心莲属的一年生植物。 [6]穿心莲茎枝呈四棱形，多分枝，质地较脆，易折断；单叶成对生长，叶柄短或近无柄，叶片展开呈披针形或卵状披针形，全缘或浅波状；上面绿色，下面灰绿色，两面光滑； [7]圆锥花序顶生或腋生花冠淡紫白色，唇形，上唇外弯，下唇直立；萌果长椭圆形至线形两侧呈压扁状，中央具一纵沟。穿心莲在5—9月开花，7—10月结果。 [8]中医五行学说认为苦人心，人只要含一小枚穿心莲的叶子，就可体会苦至心中，因此得名“穿心莲”。',
    '大青叶':'大青叶，是爵床科马蓝属多年生草本植物马蓝的叶或枝叶。 [6]其原植物马蓝根呈圆柱形； [7]茎直立或基部外倾，呈四棱柱形；叶片呈纸质，椭圆形或卵形；花冠呈现蓝紫色；蒴果呈棒状。马蓝的种子于秋季萌发出苗，越冬后于翌年早春抽茎、开花、结实，然后枯死，完成整个生长发育周期。 [8]大青叶一名，最早在《本草纲目》谓：“大青，其茎叶皆深青，故名。”',
    '大血藤':'大血藤（Sargentodoxa cuneata (Oliv.) Rehd. & E. H. Wilson in C. S. Sargent），木通科大血藤属一年生草本植物。 [5]中央小叶片菱状倒卵形至椭圆形，小叶柄短；两侧小叶较大，斜卵形，两侧不对称;花单性，雌雄异株;花萼呈长圆形，黄绿色。 [6]大血藤茎圆柱形，褐色扭曲，有条纹，砍断时有红色液汁渗出，故称“大血藤”。',
    '丹参':'丹参（Salvia miltiorrhiza Bunge）是唇形科、鼠尾草属多年生直立草本植物，根肥厚，外朱红色，内白色，肉质，叶片常为奇数羽状复叶，顶生或腋生总状花序；苞片披针形，花萼钟形，带紫色，花冠紫蓝色，花柱远外伸，小坚果黑色，椭圆形，4-8月开花，花后见果。',
    '党参':'党参（Codonopsis pilosula (Franch.) Nannf.），桔梗科党参属多年生草质藤本植物。党参的根部呈圆锥状，表面灰黄色， [7]有环纹，根的头部有凸起的茎痕和芽，习称“狮子盘头”，从中部开始有分枝； [8]根茎光滑，数量较多，都缠绕在一起； [7]叶的下端像心形，边缘是锯齿形；花为淡黄绿色，有污紫色斑点；果实的下半部分为半球形，上半部分为短圆锥形，花期7—8月，果期8—9月。 [9]党参之名始见于清代 《本草从新》 。',
    '地榆':'地榆（Sanguisorba officinalis L.）是蔷薇科地榆属多年生草本植物，高30~120厘米。茎直立，有棱。基生叶为单数羽状复叶，长椭圆形或矩圆状卵形，边缘有圆钝或波状齿;茎生叶少，小叶狭长几成长圆状披针形，顶端急尖，基部圆至心形。穗状花序密集顶生，成圆柱形或卵球形，直立;小苞片披针形，萼裂片呈花瓣状，紫红色，椭圆形，顶端常具短尖;无花瓣。瘦果褐色。花期8~9月。',
    '杜仲':'杜仲（Eucommia ulmoides Oliv.），又名胶木，为杜仲科杜仲属植物。树高可达20米，胸径约50厘米。',
    '佛手':'佛手，芸香科柑橘属常绿灌木或小乔木。其枝条有粗硬的刺；叶长椭圆形，单叶互生，革质，有腺点，有特殊的芳香气味；全年可多次开花，盛花期在4—5月间，花色以白色为主，此外还有红、紫等色； [7]果期为每年的6—10月，果实肉白，无种子。 [8]形状奇特似手，握指合拳的称“拳佛手”，伸指开展者为“开佛手”。',
    '附子':'乌头（学名：Aconitum carmichaelii Debeaux）是毛茛科、乌头属草本植物。块根倒圆锥形，茎高可达200厘米，中部之上疏被反曲的短柔毛，等距离生叶，分枝。叶片薄革质或纸质，五角形，急尖，侧全裂片不等二深裂，表面疏被短伏毛，背面通常只沿脉疏被短柔毛；叶柄疏被短柔毛。顶生总状花序；轴及花梗多少密被反曲而紧贴的短柔毛；下部苞片三裂，其他的狭卵形至披针形；小苞片生花梗中部或下部，萼片蓝紫色，外面被短柔毛，上萼片高盔形，下缘稍凹，喙不明显，花瓣无毛，雄蕊无毛或疏被短毛，子房疏或密被短柔毛，稀无毛。种子三棱形，只在二面密生横膜翅。9-10月开花。',
    '葛根':'葛根，中药名。为豆科植物野葛的干燥根，习称野葛。秋、冬二季采挖，趁鲜切成厚片或小块；干燥。甘、辛，凉。有解肌退热，透疹，生津止渴，升阳止泻，通经活络，解酒毒之功。常用于外感发热头痛，项背强痛，口渴，消渴，麻疹不透，热痢，泄泻，眩晕头痛，中风偏瘫，胸痹心痛，酒毒伤中。',
    '贯众':'贯众（Cyrtomium fortunei J. Sm.）是鳞毛蕨科，贯众属多年生蕨类植物。该种以尊敬和纪念在中国工作的苏格兰园艺学家、植物收藏家罗伯特·福特尼而命名。',
    '厚朴':'厚朴（Houpoea officinalis (Rehder & E. H. Wilson) N. H. Xia & C. Y. Wu），木兰科厚朴属落叶乔木。树皮厚，褐色，不开裂，油润而带辛辣味；叶大，集中在树枝顶部，长圆状倒卵形，下面有灰色柔毛和白粉；花白色，芳香；果实多呈长圆状卵圆形；种子三角状倒卯形；花期为5—6月，果期为8—10月。 [8]《本草纲目》记载因为木质朴而皮厚，故而叫“厚朴”。',
    '虎杖':'虎杖（Reynoutria japonica Houtt.）是蓼科，虎杖属多年生草本植物。根状茎粗壮，茎直立，高可达2米，空心，叶片宽卵形或卵状椭圆形，近革质，两面无毛，顶端渐尖，基部宽楔形、截形或近圆形，托叶鞘膜质，圆锥花序，花单性，雌雄异株，腋生；苞片漏斗状，花被淡绿色，瘦果卵形，有光泽黑褐色，8-9月开花，9-10月结果。',
    '槐花':'槐（Styphnolobium japonicum (L.) Schott），豆科槐属的落叶乔木。树皮暗灰色，树冠球形，老时则呈扁球形或倒卵形。枝叶密生，羽状复叶。圆锥花序顶生，花蝶形，夏季开黄白色花，略具芳香。荚果肉质，念珠状不开裂，黄绿色，常悬垂树梢，经冬不落，内含种子。种子肾形，棕黑色。',
    '黄柏':'黄柏（huáng bò），中药名。为芸香科植物黄皮树Phellodendron chinense Schneid.的干燥树皮。习称“川黄柏”。剥取树皮后，除去粗皮，晒干。',
    '黄精':'黄精（Polygonatum sibiricum Delar. ex Redoute），天门冬科黄精属多年生草本， [7]茎为根状；节膨大，节间一头粗、一头细，粗头有短分枝；叶4~6枚轮生，线状披针形，先端拳卷或弯曲；花序常具2—4花，成伞状，花序梗长，俯垂：苞片生于花梗基部，膜质，钻形或线状披针形，花被乳白或淡黄黄色并且筒中部稍缢缩；浆果成熟时为黑色。花期5—6月，果期8—9月。 [8]《抱朴子》中记载：“昔人以本品得坤土之气，获天地之精，故名。”',
    '黄芪':'黄芪（学名：Astragalus membranaceus (Fisch.) Bunge）是豆科、黄芪属植物。多年生草本，高50-100厘米。主根肥厚，木质，常分枝，灰白色。茎直立，上部多分枝，有细棱，被白色柔毛。羽状复叶有13-27片小叶，长5-10厘米。总状花序稍密，有10-20朵花；总花梗与叶近等长或较长，至果期显著伸长。荚果薄膜质，稍膨胀，半椭圆形，果颈超出萼外；种子3-8颗。花期6-8月，果期7-9月。',
    '金钱草':'金钱草，中药名。为报春花科植物过路黄Lysimachia christinae Hance的干燥全草。分布于云南、四川、贵州、陕西（南部）、河南、湖北、湖南、广西、广东、江西、安徽、江苏、浙江、福建等地。具有利湿退黄，利尿通淋，解毒消肿之功效。常用于湿热黄疸，胆胀胁痛，石淋，热淋，小便涩痛，痈肿疔疮，蛇虫咬伤',
    '金银花':'金银花 ，正名为忍冬（学名：Lonicera japonica Thunb. ） [1]。“金银花”一名出自《本草纲目》 [2]，由于忍冬花初开为白色，后转为黄色，因此得名金银花。药材金银花为忍冬科忍冬属植物忍冬及同属植物干燥花蕾或带初开的花。',
    '荆芥':'荆芥（Nepeta cataria L.），唇形科荆芥属多年生草本植物，全株被短柔毛；长茎为紫色的方形基部，四面有纵沟；叶成对生长，叶片分裂成三瓣；轮伞花序集生长于枝顶成假穗状；花冠唇形，青紫或淡红色；花期7—8月。果期9—10月。首载于《神农本草经》，原名“假苏”。而荆芥之名始见于《吴普本草》。',
    '决明子':'决明子是原国家卫生部公布的药食同源作物之一。 [9]2020版《中国药典》规定决明子为豆科植物钝叶决明（Senna obtusifolia (L.) H. S. Irwin & Barneby）或决明（Senna tora (L.) Roxb.）的干燥成熟种子。',
    '苦参':'苦参（Sophora flavescens Aiton），豆科槐属多年生亚灌木，高1-2米。茎直立多分枝；叶卵状披针形，背面密生柔毛；总状花序顶生，花冠黄白色，旗瓣卵状匙形；果圆柱形，呈串珠状；种子褐色扁球状；花期6-7月，果期7-9月。 [4]《本草纲目》中记载了苦参名称的由来：“苦似味名，参以功名，槐似叶形名也。”',
    '连翘':'连翘（Forsythia suspensa (Thunb.) Vahl）木樨科连翘属灌木，枝开展或下垂，棕色或淡黄褐色；叶通常为单叶，叶片呈卵形或椭圆形，先端锐尖，叶缘上面呈深绿色，下面为淡黄绿色；花通常单生或2至数朵着生于叶腋，花萼绿色，裂片呈长圆形；果呈卵球形或长椭圆形；花期3—4月；果期7—9月。 [9]因为连翘的形态如古代的连车和翘车，故得此名。',
    '络石藤':'络石（Trachelospermum jasminoides (Lindl.) Lem.），夹竹桃科络石属常绿木质藤本植物，株长达10米；小枝被短柔毛，老时无毛；叶革质，为卵形或倒卵形，具叶柄；聚伞花序圆锥状，顶生及腋生，花萼裂片窄长圆形，花冠白色；蓇葖果线状披针形；种子长圆形，顶端具白色绢毛；花期3~8月，果期6~12月。 [6]《本草纲目》记载了其名由来：“以其包络石木而生，故名络石”，是以其生长习性命名的。',
    '麦冬':'麦冬（拉丁学名：Ophiopogon japonicus (L. f.) Ker Gawl.），天门冬科沿阶草属草本植物 [9]，根较粗，中间或近末端具椭圆形或纺锤形小块根,小块根淡褐黄色；茎很短；花单生或成对生；种子球形；花期5—8月，果期8-9月。麦冬原名麦门冬，其名源于“虋冬”，最早见载于先秦著作《山海经—中山经》之条谷山：“其木多槐、桐，其草多芍药、虋冬”。',
    '墨旱莲':'墨旱莲，中药名。为菊科鳢肠属植物鳢肠Eclipta prostrata L.的干燥地上部分。分布于全国各省区。具有滋补肝肾，凉血止血之功效。常用于肝肾阴虚，牙齿松动，须发早白，眩晕耳鸣，腰膝酸软，阴虚血热、吐血、衄血、尿血，血痢，崩漏下血，外伤出血。',
    '牛滕':'为木通科植物那藤（Stauntonia hexaphylla(Thunb.)Decne.）或尾叶那藤（Stauntonia hexaphylla Decne.var.urophylla.Hand.Mazz.）的茎和根。夏、秋季采，藤茎，去枝叶；根，去须根。洗净，待润透，切段或切片，晒干。那藤生于山谷林缘或山脚灌丛中，也有栽培于庭院中。尾叶那藤多生于山坡路旁或沟谷林缘灌丛中。那藤分布于台湾、广东、广西等地。尾叶那藤分布于浙江、江西、福建、湖南、广东、广西等地。味苦,性凉。归肝、膀胱经。具有祛风散瘀、止痛、利尿消肿的功效。主治风湿痹痛、跌打伤痛、各种神经性疼痛、小便不利、水肿等病证。',
    '佩兰':'佩兰（Eupatorium fortunei Turcz.），菊科泽兰属的多年生草本植物，茎为绿色或红紫色；叶片较大，为长椭圆形或长椭圆状披针形；总苞为钟状，全部苞片为紫红色，苞片外面没有毛；花为白色或带微红色；果实为圆柱形，成熟的时候为黑褐色；冠毛白色。花、果期7—11月。屈原在《离骚》中也写道“纫秋兰以为佩”，该植物为芳香草本，香似兰花，古代妇女、儿童喜欢将其佩于身上，故名佩兰。',
    '蒲公英':'蒲公英（Taraxacum mongolicum Hand.-Mazz.）是菊科、蒲公英属多年生草本植物，叶为倒卵状披针形、倒披针形或长圆状披针形，叶柄及主脉常带红紫色；花为黄色，花的基部淡绿色，上部紫红色；内层为线状披针形；瘦果为暗褐色倒卵状披针形，冠毛为白色，长约6毫米；花期为4-9月，果期为5-10月',
    '蒲黄':'蒲黄，中药名。为香蒲科植物水烛香蒲Typha angustifolia L.、东方香蒲Typha orientalis Presl或同属植物的干燥花粉。夏季采收蒲棒上部的黄色雄花序，晒干后碾轧，筛取花粉。剪取雄花后，晒干，成为带有雄花的花粉，即为草蒲黄。',
    '前胡':'前胡（Peucedanum praeruptorum Dunn），伞形科前胡属多年生草本植物，茎干较矮，灰褐色，呈圆柱形，上面有细毛；根颈粗壮，褐色，呈圆锥形；叶子较大，呈枫叶形，表面有细毛，绿色，边缘有圆锯齿；花朵较小，淡黄色，呈多边形；花期4—6月；果期7—9月。 [4]“前胡”一名首载于《名医别录》：“前胡，苦，辛，微寒。” ',
    '肉豆蔻':'肉豆蔻（Myristica fragrans Houtt.）肉豆蔻科肉豆蔻属常绿乔木植物。肉豆蔻幼枝细长；叶片革质，椭圆形或披针形，两面光滑；花朵无毛，花被外面覆有微小绒毛，花梗比较长，脱落后残存通常为环形的疤痕；果实通常单生，有短的花柄，红色；种子卵珠形； [1]花期为9—12月；果期第二年3—6月。',
    '肉桂':'肉桂，（Cinnamomum cassia (L.) D. Don）樟科植物肉桂的干皮和枝皮，肉桂树为中等大乔木植物；树皮灰褐色，树皮上有纵向的细条纹；叶互生，叶片为长椭圆形或披针形，内卷，上面绿色，有光泽，无毛，下面淡绿色，覆盖黄色短绒毛；肉桂花花朵圆锥状，黄色；果实椭圆形，无毛；花期6—8月；果期10—12月。',
    '射干':'射干（Belamcanda chinensis (L.) Redouté）是鸢尾科射干属多年生草本植物。根状茎为不规则的块状，斜黄色或黄褐色；花橙红色，散生紫褐色的斑点；花药呈条形；子房下位，倒卵形；蒴果倒卵形或长椭圆形；种子圆球形，黑紫色，有光泽。花期6~8月，果期7~9月。 [9]射干，在《荀子·劝学篇》中就有记载：“西方有木焉，名日射干，茎长四寸，生于高山之上，而临百仞之渊。”',
    '首乌藤':'首乌藤，秋、冬二季采割，除去残叶，捆成把，干燥。性状本品呈长圆柱形，稍扭曲，具分枝，长短不一，直径3～7mm。表面紫红色至紫褐色，粗糙，具扭曲的纵皱纹。节部略膨大，有侧枝痕。外皮菲薄，可剥离。质脆，易折断，断面皮部紫红色，木部黄白色或淡棕色，导管孔明显，髓部疏松，类白色。无臭，味微苦涩。',
    '天冬':'天门冬，中药名。为百合科天门冬属植物天门冬Asparagus cochinchinensis（Lour.）Merr.的块根。植物天门冬，分布于华东、中南、西南、及河北、山西、陕西、甘肃、台湾等地。具有滋阴润燥，清肺降火之功效。主治燥热咳嗽，阴虚劳嗽，热病伤阴，内热消渴，肠燥便秘，咽喉肿痛。',
    '通草':'通脱木（Tetrapanax papyrifer (Hook.) K. Koch）是五加科通脱木属的常绿灌木或小乔木， [5]茎干粗壮，叶片宽大，基部为倒卵状长圆形或卵状长圆形，边缘全缘或疏生粗齿，侧脉和网脉不明显；叶柄粗壮，密生白色或淡棕色星状绒毛；花淡黄白色，边缘全缘或近全缘，密生白色星状绒毛，三角状卵形；果实为球形，紫黑色。花期10—12月，果期次年1—2月。 [6]现用通草为五加科植物通脱木的茎髓，古通草称为通脱木。',
    '五加皮':'五加皮，中药名。为五加科植物细柱五加Acanthopanar gracilistμlusW.W.Smith的干燥根皮。夏、秋二季采挖根部，洗净，剥取根皮，晒干。',
    '细辛':'细辛（Asarum heterotropoides F. Schmidt）是马兜铃科细辛属多年生草本植物。根细长，根状茎横走；叶卵状心形或近肾形；花紫棕色、紫褐色，花被筒壶状或半球形，内壁具纵皱褶，花被片三角状卵形，基部贴于花被筒，花丝较花药短，子房半下位或近上位，花柱柱头侧生；果半球状；花期5月， [13]果期6月。',
    '夏枯草':'夏枯草（Prunella vulgaris L.）是唇形科、夏枯草属多年生草木植物，叶为对生的卵形或椭圆状披针形，轮伞花序集成穗状；苞片肾形，顶端骤尖或尾状尖，外面和边缘有毛；花萼二唇形；花冠为紫色；小坚果为棕色。花期4~6月，果期7~10月。 [5] [7]每至夏至，夏枯草会枯黄萎谢，故名，名字最早出自《神农本草经》。',
    '仙鹤草':'龙牙草（学名：Agrimonia pilosa Ledeb. ）是蔷薇科龙牙草属的多年生草本植物，根多呈块茎状，茎的表面有稀疏柔毛；叶互生，为暗绿色，椭圆状卵形或倒卵形，有锯齿；花为穗状总状花序，花瓣为黄色，长圆形；果实为倒卵状瘦果，顶端有钩刺；花果期为5—12月。 [7]龙牙草的名字来源与其形态有关，因形似龙牙而得名。',
    '香附':'香附，中药名。为莎草科莎草属植物莎草的干燥根茎。秋季采挖，燎去毛须，置沸水中略煮或蒸透后晒干，或燎后直接晒干。',
    '小茴香':'裂叶荆芥（Schizonepeta tenuifolia (Benth.) Briq.）是唇形科、裂叶荆芥属植物。一年生草本。茎高0.3-1米，四棱形，多分枝，被灰白色疏短柔毛，茎下部的节及小枝基部通常微红色。叶通常为指状三裂，大小不等，长1-3.5厘米，宽1.5-2.5厘米。花序为多数轮伞花序组成的顶生穗状花序，长2-13厘米。花冠青紫色，长约4.5毫米，外被疏柔毛，内面无毛。小坚果长圆状三棱形，长约1.5毫米，径约0.7毫米，褐色，有小点。花期7-9月，果期在9月以后。',
    '野菊花':'野菊（Chrysanthemum indicum L.）是菊科菊属被子植物，多年生草本，有地下匍匐茎，茎枝疏被毛；中部茎叶呈卵形、长卵形或椭圆状卵形，两面淡绿色；花排成疏散伞房圆锥花序或伞房状花序，边缘白褐色，舌状花黄色，花期6~11月。',
    '益母草':'益母草（Leonurus japonicus Houtt.），属唇形科益母草属一年生或二年生草本，根上有密生须根；茎直立，钝四棱形；茎下部叶为卵形，茎中部叶为菱形；叶交互对生，有柄；叶片青绿色，质鲜嫩，揉之有汁；气微，味微苦；叶片灰绿色，多皱缩、破碎易脱落；花期6—9月，果期7—10月。 [6]益母草因其妇科多用，故有“益母”之名。',
    '菌陈':'茵陈蒿（Artemisia capillaris Thunb.），菊科蒿属的半灌木状草本植物，植株有浓香；茎直立，基部木质化，初时密生灰白色或灰黄色绢质柔毛，后渐稀疏或脱落无毛；花黄色；瘦果长圆形；花果期为秋季。茵陈蒿经冬不死，春则因陈根而生，故名因陈或茵陈，至夏其苗则变为蒿，亦称茵陈蒿，故有“三月茵陈，四月蒿，五月当柴烧”的说法。',
    '玉竹':'玉竹（Polygonatum odoratum (Mill.) Druce）是天门冬科黄精属多年生草本植物。茎圆柱形；叶互生，椭圆形或卵状矩圆形，先端尖，下面带灰白色，下面脉上平滑至呈乳头状粗糙；花被黄绿或白色，花被筒较直，花丝丝状，近平滑或具乳头状突起；浆果成熟时为蓝黑色；花期为5-6月，果期为7-9月。 [7]《本草经集注》中认为玉竹茎干强直，似竹箭杆，有节，由此得名。',
    '远志':'远志（Polygala tenuifolia Willd.），又名葽绕、蕀蒬等。产东北、华北、西北和华中以及四川；多年生草本，主根粗壮，韧皮部肉质。具有安神益智、祛痰、消肿的功能，用于心肾不交引起的失眠多梦、健忘惊悸，神志恍惚，咳痰不爽，疮疡肿毒，乳房肿痛。',
    '泽兰':'白头婆（Eupatorium japonicum Thunb. in Murray），是菊科、泽兰属的植物，分布于黑龙江、吉林、辽宁、山西、山东、河南、陕西、安徽、江苏等地区。',
    '知母':'知母（Anemarrhena asphodeloides Bunge）为天门冬科知母属多年生草本的干燥根茎。 [8]知母根状茎横生，全株无毛；叶基部丛生，呈禾叶状；总状花序，花茎直立，花被片条形，花粉红色，淡紫色至白色；蒴果六棱长卵形，种子黑色。果期6—9月。 [9]《中国药典》一书中提到春秋二季采挖，除去须根和泥沙，晒干，习称“毛知母”。',
    '枳实':'枳实，中药名。为芸香科植物酸橙Citrus aurantium L.及其栽培变种或甜橙Citrus sinensis Osbeck的干燥幼果。5～6月收集自落的果实，除去杂质，自中部横切为两半，晒干或低温干燥，较小者直接晒干或低温干燥。',
    '紫花地丁':'紫花地丁（Viola phillipina），堇菜科堇菜属多年生宿根植物。其叶片下部呈三角状卵形或狭卵形，呈长圆形、狭卵状披针形或长圆状卵形；花中等大，紫堇色或淡紫色，稀呈白色，喉部色较淡并带有紫色条纹；蒴果长圆形；种子卵球形，淡黄色；花果期4月中下旬至9月。因为其形状像一根铁钉，顶头开几朵紫花，就取了“紫花地丁”的名字。',
    '紫菀':'紫菀（Aster tataricus L. f.），菊科紫菀属多年生草本植物。茎直立，粗壮；基部叶在花期枯落，长圆状或椭圆状匙形；下部叶匙状长圆形；中部叶长圆形或长圆披针形；总苞片线形或线状披针形；花柱附片披针形；瘦果倒卵状长圆形，紫褐色。花期7~9月，果期8~10月。 [5]李时珍说，以其根色紫而柔宛，故得名“紫菀”。'
}

@app.route('/plant/<name>', methods=['GET'])
def get_plant_info(name):
    info = plant_inf.get(name, '搜索错误')
    return jsonify({'plant': name, 'info': info})

if __name__ == '__main__':
    app.run(debug=True)
