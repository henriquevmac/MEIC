namespace Assets.Scripts.IAJ.Unity.Movement.DynamicMovement
{
    public class DynamicVelocityMatch : DynamicMovement
    {
        public override string Name
        {
            get { return "VelocityMatch"; }
        }

        public float TimeToDesiredSpeed { get; set; }

        public DynamicVelocityMatch()
        {
            this.TimeToDesiredSpeed = 0.5f;
            this.Output = new MovementOutput();
        }
        public override MovementOutput GetMovement()
        {
           
            this.Output.linear = (this.Target.velocity - this.Character.Velocity)/this.TimeToDesiredSpeed;

            float maxAcceleration = Character.MaxAcceleration;
            if (this.Output.linear.sqrMagnitude > maxAcceleration*maxAcceleration)
            {
                this.Output.linear = this.Output.linear.normalized*maxAcceleration;
            }
            this.Output.angular = 0;
            return this.Output;
        }
    }
}
