FROM node:14

WORKDIR /app

COPY package*.json ./

RUN npm install --legacy-peer-deps  # 일부 버전 문제 해결 가능

COPY . .

RUN npm run build

CMD ["npm", "start"]