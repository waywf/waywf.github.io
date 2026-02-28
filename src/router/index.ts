import { createRouter, createWebHistory } from 'vue-router'
import Home from '../pages/Home.vue'
import Articles from '../pages/Articles.vue'
import ArticleDetail from '../pages/ArticleDetail.vue'
import About from '../pages/About.vue'

import NotFound from '../pages/NotFound.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
  },
  {
    path: '/articles',
    name: 'Articles',
    component: Articles,
  },
  {
    path: '/article/:id',
    name: 'ArticleDetail',
    component: ArticleDetail,
  },
  {
    path: '/about',
    name: 'About',
    component: About,
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: NotFound,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 全局前置守卫：在导航前保存文章列表的滚动位置
router.beforeEach((to, from) => {
  if (from.name === 'Articles' && to.name === 'ArticleDetail') {
    // 从文章列表进入文章详情时，保存滚动位置
    if (typeof sessionStorage !== 'undefined') {
      sessionStorage.setItem('articlesScrollPosition', window.scrollY.toString())
      console.log('Saved scroll position:', window.scrollY)
    }
  }
})

export default router
