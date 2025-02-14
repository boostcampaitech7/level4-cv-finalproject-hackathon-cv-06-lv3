import React from 'react';
import {
  Container,
  Flex,
  Heading,
  Text,
  SimpleGrid,
  Box,
  Avatar,
  Stack,
  Table,
  Thead,
  Tr,
  Th,
  Tbody,
} from '@chakra-ui/react';
import Card from 'components/Card/Card';
import CardHeader from 'components/Card/CardHeader';
import CardBody from 'components/Card/CardBody';
import TablesTableRow from 'components/Tables/TablesTableRow';

// 예시 팀원 데이터
const tablesTableData = [
  {
    logo: 'https://avatars.githubusercontent.com/u/38092232?v=4',
    name: '김태성',
    email: 'randfo42@gmail.com',
    subdomain:
      '문제정의, Surrogate Model, Search Model 템플릿 작성 및 문제 구성, 모델링-UI 연결, UI 요구 사항 구현',
    domain: 'AI Modeling',
    status: 'https://github.com/randfo42',
    notion:
      'https://wise-columnist-5eb.notion.site/162218e09e1680ada4e2cb13e50ae624?pvs=4',
  },
  {
    logo: 'https://avatars.githubusercontent.com/u/174946708?v=4',
    name: '김경윤',
    email: 'ruddbs803@naver.com',
    subdomain:
      'GA 구현, GA(니치함수) 실험 , Surrogate - Search 연결, Evaluation Metric 구현\nSurrogate(TabPFN, CatBoost)구현, Surrogate Model 고도화 및 테스트',
    domain: 'AI Modeling',
    status: 'https://github.com/kkyungyoon',
    notion:
      'https://wise-columnist-5eb.notion.site/162218e09e1680748f09e6ffb8f9c4a1?pvs=4',
  },
  {
    logo: 'https://avatars.githubusercontent.com/u/91772398?v=4',
    name: '김영석',
    email: 'rpfps3232@naver.com',
    subdomain:
      'RL을 응용한 search 모델 작성, 테스트 코드 정리, 데이터셋 별 테스트 후 오류 발견 및 공유 또는 수정',
    domain: 'AI Modeling',
    status: 'https://github.com/kimyoungseok3232',
    notion:
      'https://wise-columnist-5eb.notion.site/162218e09e16802199b8f0f8cc217838?pvs=4',
  },
  {
    logo: 'https://avatars.githubusercontent.com/u/105529609?v=4',
    name: '박태영',
    email: 'mu07010@naver.com',
    subdomain:
      '요구사항 화면 분석과 기능설계도 및 DB설계 ERD 작성, DjangoRestframework를 이용한 RESTful API 구축',
    domain: 'BackEnd',
    status: 'https://github.com/taeyoung1005',
    notion:
      'https://wise-columnist-5eb.notion.site/162218e09e16813aa477dcfbf636a6bb?pvs=4',
  },
  {
    logo: 'https://avatars.githubusercontent.com/u/127745394?v=4',
    name: '신영태',
    email: 'youngtae0818@naver.com',
    subdomain:
      '데이터 자동화 전처리 코드 구현 및 대용량 데이터 최적화, 효율적인 데이터 병합 및 전처리 파이프라인 구축\nEDA 및 feature engineering, target 변수 최적화',
    domain: 'Data Analyze',
    status: 'https://github.com/Dangtae',
    notion:
      'https://wise-columnist-5eb.notion.site/162218e09e16804ab549cf8c7e3aba80?pvs=4',
  },
  {
    logo: 'https://avatars.githubusercontent.com/u/113595155?v=4',
    name: '함로운',
    email: 'kb721a@naver.com',
    subdomain: 'React & Chakra UI를 활용한 웹 기반 User-interface 개발',
    domain: 'FrontEnd',
    status: 'https://github.com/andantecode',
    notion:
      'https://wise-columnist-5eb.notion.site/162218e09e1680faa29dc7a9702145bc?pvs=4',
  },
];

const About = () => {
  return (
    <Container maxW="container.xl" py={8}>
      {/* 헤더 영역 */}
      <Flex direction="column" align="center" textAlign="center" mb={0}>
        <Heading as="h1" size="2xl" mb={0} color="#fff">
          Meet Our Team
        </Heading>
      </Flex>

      <Flex direction="column" pt={{ base: '120px', md: '75px' }}>
        {/* Authors Table */}
        <Card overflowX={{ sm: 'scroll', xl: 'hidden' }} pb="0px">
          <CardHeader p="4px 0px 20px 0px">
            <Text fontSize="lg" color="#fff" fontWeight="bold">
              Authors Table
            </Text>
          </CardHeader>
          <CardBody>
            <Table variant="simple" color="#fff">
              <Thead>
                <Tr my=".8rem" ps="0px" color="gray.400">
                  <Th
                    ps="0px"
                    color="gray.400"
                    fontFamily="Plus Jakarta Display"
                    borderBottomColor="#56577A"
                  >
                    Author
                  </Th>
                  <Th
                    color="gray.400"
                    fontFamily="Plus Jakarta Display"
                    borderBottomColor="#56577A"
                  >
                    Role
                  </Th>
                  <Th
                    color="gray.400"
                    fontFamily="Plus Jakarta Display"
                    borderBottomColor="#56577A"
                  >
                    Links
                  </Th>
                </Tr>
              </Thead>
              <Tbody>
                {tablesTableData.map((row, index, arr) => {
                  return (
                    <TablesTableRow
                      name={row.name}
                      logo={row.logo}
                      email={row.email}
                      subdomain={row.subdomain}
                      domain={row.domain}
                      status={row.status}
                      notion={row.notion}
                      lastItem={index === arr.length - 1 ? true : false}
                    />
                  );
                })}
              </Tbody>
            </Table>
          </CardBody>
        </Card>
      </Flex>
    </Container>
  );
};

export default About;
