import bgBody from 'assets/img/background-body-admin.png';

export const bgAdmin = {
  styles: {
    global: props => ({
      '*': {
        userSelect: 'none',
      },
      html: {
        msOverflowStyle: 'none', // IE, Edge
        scrollbarWidth: 'none', // Firefox
        '::-webkit-scrollbar': {
          display: 'none', // Chrome, Safari, Opera
        },
      },
      body: {
        bgImage: bgBody,
        bgSize: 'cover',
        bgPosition: 'center center',
        overflowY: 'scroll', // 스크롤 가능하게 둬야함
        // 웹킷 브라우저용 (여기서 width: 0 은 안 먹을 수 있으니 display: none 사용)
        '::-webkit-scrollbar': {
          display: 'none',
        },
      },
    }),
  },
};
