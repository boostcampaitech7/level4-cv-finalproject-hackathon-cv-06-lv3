import React from 'react';
import {
  Box,
  Flex,
  Text,
  Button,
  Container,
  Heading,
  SimpleGrid,
  useColorModeValue,
} from '@chakra-ui/react';
import { Link } from 'react-router-dom';
import Card from 'components/Card/Card';
import CardHeader from 'components/Card/CardHeader';
import CardBody from 'components/Card/CardBody';
import { Separator } from 'components/Separator/Separator';

const features = [
  {
    title: 'Actionable Predictions',
    description:
      'Not just predicting outcomes Argmax mini prescribes optimal actions to drive better decisions.',
  },
  {
    title: 'Domain-Robust Performance',
    description:
      'Engineered to operate reliably across a diverse range of datasets, our system delivers robust recommendations regardless of domain-specific challenges.',
  },
  {
    title: 'Flexible Multi-Objective Optimization',
    description:
      'Within a single dataset, our AI dynamically adapts to multiple optimization goals, offering tailored recommendations that align with varying strategic objectives.',
  },
];

const processSteps = [
  {
    title: 'Data Upload & Selection',
    description:
      'Users upload various datasets and select the ones needed for analysis.',
  },
  {
    title: 'Data Verification',
    description:
      'Review the selected datasets to ensure quality and consistency before analysis.',
  },
  {
    title: 'Set Optimization Goals',
    description:
      'Define and prioritize optimization targets to guide data-driven decision-making.',
  },
  {
    title: 'Result Evaluation',
    description:
      'Examine prediction outcomes and recommendations to derive actionable strategies.',
  },
];

function MainPage() {
  // Hero section background: use a full-width background image.

  return (
    <Container maxW="container.xl" py={10}>
      {/* Hero Section */}
      <Flex
        direction="column"
        align="center"
        textAlign="center"
        py={8}
        px={8}
        mb={10}
        position="relative"
        overflow="hidden"
      >
        {/* Optional overlay to enhance text readability */}
        <Box
          position="absolute"
          top="0"
          left="0"
          width="100%"
          height="100%"
          zIndex={0}
        />
        <Heading
          as="h1"
          size="2xl"
          mb={4}
          color="white"
          textShadow="0 2px 10px rgba(0, 0, 0, 0.6)"
          zIndex={1}
        >
          SIXSENSE mini
        </Heading>
        <Text
          fontSize="xl"
          mb={6}
          color="white"
          textShadow="0 2px 10px rgba(0, 0, 0, 0.6)"
          zIndex={1}
        >
          Prescriptive AI for Actionable Recommendations
        </Text>
      </Flex>

      {/* Features Section */}
      <Box mb={10}>
        <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8}>
          {features.map((feature, index) => (
            <Card
              key={index}
              p={6}
              boxShadow="md"
              transition="all 0.3s ease-in-out"
              _hover={{
                boxShadow: 'xl',
                transform: 'translateY(-10px)',
              }}
            >
              <CardHeader mb={4}>
                <Text fontSize="xl" color="brand.100" fontWeight="bold">
                  {feature.title}
                </Text>
              </CardHeader>
              <Separator mb={4} />
              <CardBody>
                <Text fontSize="md" color="gray.200">
                  {feature.description}
                </Text>
              </CardBody>
            </Card>
          ))}
        </SimpleGrid>
      </Box>

      {/* Process Flow Section */}
      <Box pt={10}>
        <SimpleGrid columns={{ base: 1, md: 4 }} spacing={8}>
          {processSteps.map((step, index) => (
            <Card
              key={index}
              p={6}
              boxShadow="md"
              transition="all 0.3s ease-in-out"
              _hover={{
                boxShadow: 'xl',
                transform: 'translateY(-10px)',
              }}
            >
              <CardHeader mb={4}>
                <Text fontSize="lg" fontWeight="bold" color="blue.300">
                  {step.title}
                </Text>
              </CardHeader>
              <Separator mb={4} />
              <CardBody>
                <Text fontSize="sm" color="gray.600">
                  {step.description}
                </Text>
              </CardBody>
            </Card>
          ))}
        </SimpleGrid>
      </Box>
    </Container>
  );
}

export default MainPage;
