import requests
import json

def main():
    url = "https://imagesearch.backend.mercor.io/rank_images"
    body = {
  "query":"https://nb.scene7.com/is/image/NB/bbw550bb_nb_02_i?$dw_detail_main_lg$&bgc=f1f1f1&layer=1&bgcolor=f1f1f1&blendMode=mult&scale=10&wid=1600&hei=1600",
  "links":[
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/07/64081c20932a8aa079f32f61/s_wp_64081c3917fb4b42dbfab2b1.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/07/15/62d22653c3c2d055467a333c/s_wp_62d226bc6f6c91fff0baac75.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/06/64067df5b0f193e85ef2a7b2/s_wp_64067e6858083db0cf958e5d.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/06/63e1720e4bd7606a15bca3e6/s_wp_63e1722f7dfcc24d72983dea.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/14/6410f9385d686b27c9707c09/s_wp_6410f9451741be954d848a94.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/01/16/63c5e52fb0f1935864f64b92/s_wp_63dea4d832c1dc1bb5da14bf.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/11/63e8081edbb0e76416886af6/s_63e809a987a2f545a83411db.jpg",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/19/64409a1687a2f597e0f58704/s_wp_64409a1b87a2f597e0f5872a.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/01/642855db528015e8491ef575/s_wp_642855dd3b982ad0e2f5e43c.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/30/64265e89c9a2282811371a71/s_wp_643433972fbf1a6c44163359.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/01/22/63cdbe994bf9ff3c48c1b7db/s_wp_63cdbebd91e053f47544dc3e.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/23/63f828de9464f3208614988a/s_wp_63f8da87046d74bd139ffc84.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/12/12/6397978d046d744529e66c20/s_63ae12664bf9ffcbb56918b9.jpg",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/14/64396cf14bf9ff4d6fae5859/s_wp_64396d07932a8afbc36d996d.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/12/64370aeb1b40a78c2e9ff550/s_64419750dbb0e789fd992041.jpeg",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/08/640941203e37e3a6e034ad11/s_wp_640944a9acf462b7e5c18390.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/10/24/6356f4d3ff04843d4873a184/s_wp_635700ef97b5d0d77afd186a.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/16/643ca22902760bc4ffec756e/s_wp_643ca26483cbecc57704a79a.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/16/641315070b299fd856c08cd1/s_wp_641315070b299fd856c08cd2.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/27/63fcbb36e5460fc649a32f96/s_wp_63fcbb36e5460fc649a32f97.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/29/6424ed0affb5d0c429c9b1df/s_wp_6424ed1204166d964366eac9.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/12/04/638d5715beca4e98d4a05ed8/s_639d150517fb4b041361dfa9.jpg",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/03/642bc02381078af34ca1ce91/s_wp_642bffc6a0aeb77f4bc93c9e.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/31/6426b60d0382914b7cc53dab/s_wp_6426b60d0382914b7cc53dac.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/17/643d6e6b5d686b4124c0d76e/s_64404b3358083d1213cf13c3.jpg",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/13/63ea13a34bf9ffe2f4d8871b/s_wp_63ea13b9eb7e7aa91bd432f0.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/20/63f43ca602760b2a7bd68bef/s_wp_63f43cc9bd06293bc66e36bc.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/26/6420cc346665f375a9f8c604/s_wp_6420cc37678c3ae303e26545.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/24/63f9619484b307fd45ab14a6/s_63f961ac02760b16342041d1.jpg",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/04/6403ec1f02760b5a48a926e5/s_wp_6403ec2d92e491b1db257dc0.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/09/64337e4f694d7223a363fd4b/s_wp_64337f9756b2f8115c57d0f9.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/01/19/63c9b7754bc655c915be0cee/s_wp_63c9b7a9fed51f19d74ceda0.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/06/15/62aa4d2e21f2e1aadd6de75c/s_wp_62aa4d48008b99fa8f8d4636.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/06/15/62aa4c635d7f8712c17b3d32/s_wp_62aa4c7dcf7b27acc53992b3.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/14/643974d5dbb0e792a2cf9ac2/s_wp_643974ed83cbeca29dee6995.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/13/63eaa35756b2f81eaf30a1e0/s_wp_63eaa37ec1c346a5e236165f.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/11/01/6361cd643b982af4ae8abbd2/s_wp_6361cda232c1dc075a8fcd20.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/03/63dd8ca40a6a865f1b71dcc7/s_63dd8f678bb2e2894efb9787.jpeg",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/12/17/639e465687a2f5cb01a9bf0b/s_wp_639e4668c1c3468ee92e55ee.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/07/64073a8087a2f5a0fa939e79/s_wp_64073a8087a2f5a0fa939e7a.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/02/642a1333dbb0e79c170c3d85/s_wp_642a1333dbb0e79c170c3d86.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/02/17/63efd8bda0aeb7ffc969f78e/s_63efdea2bd06290d344e786a.jpg",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/11/6435972e34e25350b78c6c52/s_wp_6435972f253a8c39468ff4e4.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/06/10/62a3d79eefd0e4c30b04c432/s_wp_62a3d7c6bcbb524a23821ca5.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/26/6420fbf49464f3ddea8479ad/s_wp_6420fc20ff04842d67cf692f.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/05/64046bdce5b9cf9227da9cad/s_wp_64046eef81078a9c233a5f7d.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/05/642d32abe94898fc647f8ad6/s_6440f84217e49c2d79247787.jpeg",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/04/05/642e0885678c3a5f8847afd0/s_wp_642e08a0c5df6ce1f99f8c81.webp",
"https://images1.vinted.net/t/03_00b3e_FewJGoxrfLNd8LyHub54XPtP/f800/1681777028.jpeg?s=a3c30d34b3a6fe38c1b1a8e408ac27d64dc7b0ea" ]}

    headers = {
        'Content-Type': 'application/json'
    }
    res = requests.post(url, data=json.dumps(body), headers=headers)
    print(res.json())


if __name__ == "__main__":
    main()