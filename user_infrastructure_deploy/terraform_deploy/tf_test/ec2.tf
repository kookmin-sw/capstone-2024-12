resource "aws_security_group" "instance_sg" {
  ingress = [{
    cidr_blocks      = ["0.0.0.0/0"]
    description      = ""
    from_port        = 22
    ipv6_cidr_blocks = []
    prefix_list_ids  = []
    protocol         = "tcp"
    security_groups  = []
    self             = false
    to_port          = 22
  }]
  egress = [{
    cidr_blocks      = ["0.0.0.0/0"]
    description      = ""
    from_port        = 0
    ipv6_cidr_blocks = []
    prefix_list_ids  = []
    protocol         = "-1"
    security_groups  = []
    self             = false
    to_port          = 0
  }]

  vpc_id = aws_vpc.vpc.id
  
  tags = {
    "Name" = "mjh-instance-sg"
  }
}

resource "aws_instance" "instance" {
  ami = "ami-0f3a440bbcff3d043"
  instance_type = "t3.micro"
  subnet_id = aws_subnet.subnet_a.id
  vpc_security_group_ids = [aws_security_group.instance_sg.id]
  tags = {
    "Name" = "mjh-instance"
  }
}
